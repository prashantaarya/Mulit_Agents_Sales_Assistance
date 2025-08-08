# tools/hybrid_search.py

import pandas as pd
import ast
from typing import Dict, List, Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.docstore.document import Document
from typing import Literal

# Pydantic models for dynamic filter generation
class FilterCondition(BaseModel):
    field: str = Field(..., description="Column name to filter on")
    operator: Literal['contains', 'not_contains', 'equals', 'not_equals'] = Field(..., description="Filter operator - only use basic operators")
    value: str = Field(..., description="Value to search for")

class FilterList(BaseModel):
    filters: List[FilterCondition]

class HybridSearchToolBox:
    def __init__(self, df: pd.DataFrame, llm):
        self.df = df
        self.llm = llm
        self.column_descriptions = {
            'Prospect Business Name': 'Business name',
            'Primary Category': 'Main industry (e.g., Computer Contractors)',
            'City': 'Business location city', 
            'State': 'Business location state',
            'BuzzBoard Data Parsed': 'Contains digital marketing signals like Google Places, SEM, social media activity'
        }
        self._create_content_vector_store()
        self._create_extractor_chain()

    def _create_content_vector_store(self):
        print("Creating content vector store...")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        documents = []
        for index, row in self.df.iterrows():
            content = (
                f"Business: {row.get('Prospect Business Name', '')}. "
                f"Category: {row.get('Primary Category', '')}. "
                f"Location: {row.get('City', '')}, {row.get('State', '')}"
            )
            documents.append(Document(page_content=content, metadata={"index": index}))
        
        self.content_vector_store = FAISS.from_documents(documents, embedding_model)
        print("Content vector store created.")

    def _create_extractor_chain(self):
        parser = self.llm.with_structured_output(FilterList)
        prompt = ChatPromptTemplate.from_template(
            "Extract ONLY basic demographic filters. Do NOT extract complex concepts like 'low presence' or 'high spend'.\n"
            "Available fields:\n{columns}\n\n"
            "For categories, use 'contains' operator to catch variations:\n"
            "- 'Computer Contractors' → field='Primary Category', operator='contains', value='Computer'\n"
            "- 'IT Services' → field='Primary Category', operator='contains', value='IT'\n"
            "- 'businesses in Texas' → field='State', operator='equals', value='TX'\n\n"
            "IGNORE complex terms like: low/high presence, ads spend, digital gaps\n"
            "Query: {query}"
        )
        self.extractor_chain = prompt | parser

    def find_prospects_hybrid(self, query: str) -> List[Dict]:
        """
        Run intelligent hybrid filtering on the dataset using extracted filters and fuzzy logic.
        
        Args:
            query (str): Natural language query describing the prospects you're looking for
            
        Returns:
            List[Dict]: List of prospect dictionaries matching the criteria
        """
        print(f" Processing query: '{query}'")

        # Step 1: Extract basic filters from the query
        column_info = "\n".join([f"- {name}: {desc}" for name, desc in self.column_descriptions.items()])
        try:
            filter_list_obj = self.extractor_chain.invoke({"query": query, "columns": column_info})
            filters = filter_list_obj.filters
        except Exception as e:
            print(f"Error extracting filters: {e}")
            return [{"Status": "Failed to process query"}]

        print(f" Extracted Filters: {[(f.field, f.operator, f.value) for f in filters]}")
        filtered_df = self.df.copy()

        # Step 2: Apply filters with basic and fuzzy matching
        for f in filters:
            if f.field not in filtered_df.columns:
                print(f"Column '{f.field}' not found in DataFrame. Skipping.")
                continue

            col = filtered_df[f.field].astype(str).str.lower()
            val = str(f.value).lower()

            if f.operator == 'contains':
                filtered_df = filtered_df[col.str.contains(val, na=False, case=False)]
            elif f.operator == 'equals':
                exact_match = filtered_df[col == val]
                if exact_match.empty and f.field == 'Primary Category':
                    fuzzy_match = filtered_df[col.str.contains(val, na=False, case=False)]
                    if not fuzzy_match.empty:
                        print(f"Fuzzy matched '{val}' in '{f.field}': {fuzzy_match[f.field].unique()[:3]}")
                        filtered_df = fuzzy_match
                    else:
                        filtered_df = exact_match
                else:
                    filtered_df = exact_match
            elif f.operator == 'not_contains':
                filtered_df = filtered_df[~col.str.contains(val, na=False, case=False)]
            elif f.operator == 'not_equals':
                filtered_df = filtered_df[col != val]

        print(f"Prospects after filtering: {len(filtered_df)}")

        # Step 3: If no results, fallback to broader search
        if filtered_df.empty:
            print(" No results found — attempting broader search.")
            return self._try_broader_search(query)

        # Step 4: Apply business intelligence scoring
        results = []
        for _, row in filtered_df.iterrows():
            analysis = self._analyze_prospect(row, query)
            if analysis.get('relevance_score', 0) > 0:
                results.append(analysis)

        # Step 5: Fallback if no high relevance results
        if not results:
            print("No high-relevance matches — returning fallback results.")
            for _, row in filtered_df.head(5).iterrows():
                fallback_analysis = self._analyze_prospect(row, query, relaxed=True)
                results.append(fallback_analysis)

        # Step 6: Return top sorted results
        results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        top_results = results[:10] if results else [{"Status": "No prospects found matching criteria"}]

        print(f"Returning {len(top_results)} prospect(s)")
        return top_results

    def _try_broader_search(self, query: str) -> List[Dict]:
        """Fallback search with broader criteria"""
        print("Trying broader search...")
        
        # Extract key terms from query
        query_lower = query.lower()
        broader_results = []
        
        # Try category variations
        if 'computer' in query_lower:
            computer_matches = self.df[self.df['Primary Category'].str.contains('Computer', case=False, na=False)]
            if not computer_matches.empty:
                print(f"Found {len(computer_matches)} prospects with 'Computer' in category")
                for _, row in computer_matches.head(5).iterrows():
                    broader_results.append(self._analyze_prospect(row, query, relaxed=True))
        
        # Try contractor variations
        if 'contractor' in query_lower:
            contractor_matches = self.df[self.df['Primary Category'].str.contains('Contract', case=False, na=False)]
            if not contractor_matches.empty:
                print(f"Found {len(contractor_matches)} prospects with 'Contract' in category")
                for _, row in contractor_matches.head(5).iterrows():
                    broader_results.append(self._analyze_prospect(row, query, relaxed=True))
        
        if broader_results:
            return broader_results[:10]
        else:
            # Show sample categories available
            sample_categories = self.df['Primary Category'].value_counts().head(10)
            return [{
                "Status": "No exact matches found",
                "Suggestion": "Try these available categories",
                "Available Categories": sample_categories.to_dict()
            }]

    def _analyze_prospect(self, row, original_query: str, relaxed: bool = False) -> Dict:
        """Analyze prospect against original query requirements"""
        buzzboard = row.get('BuzzBoard Data Parsed', {})
        if isinstance(buzzboard, str):
            try:
                buzzboard = ast.literal_eval(buzzboard) if buzzboard != 'Not Found' else {}
            except:
                buzzboard = {}
        
        # Calculate digital presence scores
        local_presence_score = 0
        if buzzboard.get('Google Places') == 'Yes': local_presence_score += 3
        if isinstance(buzzboard.get('Reviews (local and social)'), (int, float)) and buzzboard.get('Reviews (local and social)', 0) > 10: local_presence_score += 2
        if buzzboard.get('FB latest_posts') == 'Yes': local_presence_score += 1
        
        sem_score = 3 if buzzboard.get('SEM') == 'Yes' else 0
        
        # Determine relevance based on query
        relevance_score = 1  # Base score
        gaps = []
        opportunities = []
        
        # Check for "low local presence" requirement
        if 'low local presence' in original_query.lower() or 'weak local presence' in original_query.lower():
            if local_presence_score <= 2:
                relevance_score += 3
                gaps.append('Weak Local Presence')
                opportunities.append('Improve Google My Business & Local SEO')
            elif not relaxed:
                relevance_score = 0  # Doesn't match requirement in strict mode
        
        # FIX: Check for "high google ads" or "high SEM" requirement  
        if any(term in original_query.lower() for term in ['high google ads', 'high sem', 'google ads spend', 'high ads spend']):
            if sem_score >= 3:  # Has SEM activity
                relevance_score += 3
                opportunities.append('Optimize Current SEM Campaigns')
            else:
                # If query specifically asks for high SEM but business has none, exclude it
                if not relaxed:
                    relevance_score = 0  # EXCLUDE businesses with no SEM activity
                    return {
                        'Prospect Business Name': row.get('Prospect Business Name'),
                        'Status': 'Filtered out - No Google Ads activity found',
                        'relevance_score': 0
                    }
        
        # Add general gaps
        if buzzboard.get('Google Places', 'No') == 'No': 
            gaps.append('No Google Places Listing')
        if buzzboard.get('SEM', 'No') == 'No': 
            gaps.append('No Paid Search Activity')
        if buzzboard.get('FB latest_posts', 'No') == 'No': 
            gaps.append('Inactive Social Media')
        
        # In relaxed mode, give points for any gaps (business opportunities)
        if relaxed and gaps:
            relevance_score += len(gaps)
        
        return {
            'Prospect Business Name': row.get('Prospect Business Name'),
            'Primary Category': row.get('Primary Category'),
            'Location': f"{row.get('City', '')}, {row.get('State', '')}",
            'Local Presence Score': f"{local_presence_score}/6",
            'SEM Activity': 'Active' if sem_score > 0 else 'None',
            'Key Gaps': '; '.join(gaps) if gaps else 'Strong Digital Presence',
            'Opportunities': '; '.join(opportunities) if opportunities else 'Maintain Current Strategy',
            'Match Type': 'Relaxed' if relaxed else 'Strict',
            'relevance_score': relevance_score
        }
    # Update get_prospect_details to include timing
    def get_prospect_details(self, prospect_name: str) -> Optional[Dict]:
        """Get detailed prospect analysis with SWOT, trends, and timing"""
        match = self.df[self.df['Prospect Business Name'].str.lower() == prospect_name.lower()]
        if not match.empty:
            prospect_data = match.iloc[0]
            buzzboard = prospect_data.get('BuzzBoard Data Parsed', {})
            if isinstance(buzzboard, str):
                try:
                    buzzboard = ast.literal_eval(buzzboard) if buzzboard != 'Not Found' else {}
                except:
                    buzzboard = {}
            
            # Calculate scores
            seo_score = self._calculate_seo_score(buzzboard)
            social_score = self._calculate_social_score(buzzboard)
            d_score = (seo_score + social_score) / 2
            
            # Generate SWOT analysis
            swot = self._generate_swot_analysis(buzzboard, prospect_data)
            
            # Add market trends and timing
            market_analysis = self._analyze_market_trends(prospect_data)
            timing_strategy = self._get_optimal_timing(prospect_data, swot)
            
            return {
                "Prospect Business Name": prospect_data.get("Prospect Business Name"),
                "Primary Category": prospect_data.get("Primary Category"),
                "Location": f"{prospect_data.get('City', '')}, {prospect_data.get('State', '')}",
                "BuzzBoard Metrics": {
                    "D-Score": f"{d_score:.1f}/10",
                    "SEO Score": f"{seo_score}/10",
                    "Social Media Score": f"{social_score}/10"
                },
                "SWOT Analysis": swot,
                "Market Trends": market_analysis["trends"],
                "Competitor Analysis": market_analysis["competitors"],
                "Communication Timing": timing_strategy,
                "Engagement Strategy": self._get_engagement_strategy(swot)
            }
        return None
    
    def _calculate_seo_score(self, buzzboard: dict) -> int:
        """Calculate SEO score from BuzzBoard data"""
        score = 0
        if buzzboard.get('Google Places') == 'Yes': score += 4
        if buzzboard.get('SEM') == 'Yes': score += 3
        reviews = buzzboard.get('Reviews (local and social)', 0)
        if isinstance(reviews, (int, float)) and reviews > 20: score += 3
        return min(score, 10)

    def _calculate_social_score(self, buzzboard: dict) -> int:
        """Calculate social media score"""
        score = 0
        if buzzboard.get('FB latest_posts') == 'Yes': score += 5
        if buzzboard.get('Instagram') == 'Yes': score += 3
        if buzzboard.get('Twitter') == 'Yes': score += 2
        return min(score, 10)

    def _generate_swot_analysis(self, buzzboard: dict, prospect_data) -> dict:
        """Generate SWOT analysis"""
        strengths, weaknesses, opportunities = [], [], []
        
        # Strengths
        if buzzboard.get('Google Places') == 'Yes':
            strengths.append("Strong local presence with Google Places listing")
        if buzzboard.get('SEM') == 'Yes':
            strengths.append("Active in paid search advertising")
        if buzzboard.get('FB latest_posts') == 'Yes':
            strengths.append("Maintains active social media presence")
        
        # Weaknesses
        if buzzboard.get('Google Places') != 'Yes':
            weaknesses.append("Missing Google Places listing - invisible in local searches")
        if buzzboard.get('SEM') != 'Yes':
            weaknesses.append("No paid search presence - missing potential leads")
        if buzzboard.get('FB latest_posts') != 'Yes':
            weaknesses.append("Inactive social media - poor customer engagement")
        
        # Opportunities
        if not strengths:
            opportunities.append("Complete digital transformation opportunity")
        if buzzboard.get('Google Places') != 'Yes':
            opportunities.append("Establish Google My Business for local visibility")
        if buzzboard.get('SEM') != 'Yes':
            opportunities.append("Launch targeted Google Ads campaigns")
        
        return {
            "Strengths": strengths if strengths else ["Limited digital presence"],
            "Weaknesses": weaknesses,
            "Opportunities": opportunities,
            "Threats": ["Competitors with stronger digital presence"]
        }

    def _get_engagement_strategy(self, swot: dict) -> str:
        """Generate personalized engagement strategy"""
        if len(swot['Weaknesses']) >= 3:
            return "Lead with comprehensive digital marketing audit. Focus on immediate wins like Google My Business setup."
        elif 'Google Places' in str(swot['Weaknesses']):
            return "Start conversation around local SEO and Google My Business optimization."
        elif 'SEM' in str(swot['Weaknesses']):
            return "Position paid search solutions to capture competitor traffic."
        else:
            return "Focus on optimization and advanced digital marketing strategies."
        
    
    # Add these methods to HybridSearchToolBox class in tools/hybrid_search.py

    def _analyze_market_trends(self, prospect_data) -> dict:
        """Analyze market trends and competitor landscape"""
        category = prospect_data.get('Primary Category', '')
        location = prospect_data.get('State', '')
        
        # Get similar businesses in same category/location
        similar_businesses = self.df[
            (self.df['Primary Category'] == category) & 
            (self.df['State'] == location)
        ]
        
        # Calculate market benchmarks
        total_similar = len(similar_businesses)
        
        if total_similar <= 1:
            return {
                "trends": ["Limited market data available"],
                "competitors": ["Market analysis needs more data"]
            }
        
        # Digital presence trends
        google_places_adoption = 0
        sem_adoption = 0
        social_adoption = 0
        
        for _, row in similar_businesses.iterrows():
            buzzboard = row.get('BuzzBoard Data Parsed', {})
            if isinstance(buzzboard, str):
                try:
                    buzzboard = ast.literal_eval(buzzboard) if buzzboard != 'Not Found' else {}
                except:
                    buzzboard = {}
            
            if buzzboard.get('Google Places') == 'Yes': google_places_adoption += 1
            if buzzboard.get('SEM') == 'Yes': sem_adoption += 1
            if buzzboard.get('FB latest_posts') == 'Yes': social_adoption += 1
        
        # Calculate percentages
        gp_percent = (google_places_adoption / total_similar * 100)
        sem_percent = (sem_adoption / total_similar * 100)
        social_percent = (social_adoption / total_similar * 100)
        
        trends = []
        if gp_percent > 70: trends.append(f"High Google Places adoption ({gp_percent:.0f}%) in {category}")
        if sem_percent > 50: trends.append(f"Growing SEM usage ({sem_percent:.0f}%) in local market")
        if social_percent < 30: trends.append(f"Low social media presence ({social_percent:.0f}%) - opportunity area")
        
        # Competitor insights
        competitors = [f"{total_similar-1} similar businesses in {location}"]
        if gp_percent > 80: competitors.append("Most competitors have strong local SEO presence")
        if sem_percent > 60: competitors.append("Majority using paid search - competitive landscape")
            
        return {
            "trends": trends if trends else ["Market shows standard digital adoption"],
            "competitors": competitors
        }

    def _get_optimal_timing(self, prospect_data, swot_analysis) -> dict:
        """Generate optimal communication timing suggestions"""
        category = prospect_data.get('Primary Category', '').lower()
        gaps_count = len(swot_analysis.get('Weaknesses', []))
        
        # Industry-based timing
        if any(word in category for word in ['computer', 'it', 'tech', 'software']):
            best_days = "Tuesday-Thursday"
            best_time = "10am-2pm EST"
            reason = "B2B tech businesses most responsive mid-week"
        elif any(word in category for word in ['contractor', 'plumbing', 'electrical']):
            best_days = "Monday-Wednesday" 
            best_time = "8am-11am EST"
            reason = "Service businesses check emails early morning"
        else:
            best_days = "Tuesday-Thursday"
            best_time = "10am-3pm EST"
            reason = "Standard business hours"
        
        # Urgency-based timing
        if gaps_count >= 3:
            urgency = "Send immediately"
            follow_up = "Follow up in 3-5 days"
        elif gaps_count >= 2:
            urgency = "Send within 24 hours"
            follow_up = "Follow up in 1 week"
        else:
            urgency = "Send within 2-3 days"
            follow_up = "Follow up in 2 weeks"
        
        return {
            "optimal_timing": f"{best_days}, {best_time} - {reason}",
            "urgency": urgency,
            "follow_up_schedule": follow_up
        }
            