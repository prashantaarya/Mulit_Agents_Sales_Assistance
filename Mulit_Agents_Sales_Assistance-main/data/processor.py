# data/processor.py
import pandas as pd
import ast

def process_data(file_path: str, sheet_name: str = 'Data', header_row: int = 1) -> pd.DataFrame:
    """Clean and process the Excel data for agent consumption."""
    try:
        # Load the dataset from the specified sheet and header
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row)
        print("Excel file loaded successfully.")

        # Clean column names
        df.columns = df.columns.str.strip().str.replace('\n', ' ').str.replace('\xa0', ' ')

        # CRITICAL FIX: Use the correct column name from the file
        column_mapping = {
            'User Name': 'Sales Rep Name',
            'Business Name': 'Prospect Business Name',
            'Address': 'Prospect Address',
            'Category - Primary': 'Primary Category',
            'Category - Secondary': 'Secondary Category',
            'All Signals/SMB Data Points': 'BuzzBoard Data', # Corrected column name
            'Products': 'Products Sold'
        }

        # Apply mapping only for columns that exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df[new_name] = df[old_name]

        # Forward fill essential columns
        if 'Customer' in df.columns: df['Customer'] = df['Customer'].ffill()
        if 'Products Sold' in df.columns: df['Products Sold'] = df['Products Sold'].ffill()

        # Drop irrelevant rows
        if 'UID' in df.columns: df.dropna(subset=['UID'], inplace=True)
        if 'Prospect Business Name' in df.columns:
            df = df[df['Prospect Business Name'] != 'SMB'].copy()

        # Parse BuzzBoard Data with better error handling
        def safe_parse_buzzboard(data_string):
            if pd.isna(data_string) or str(data_string).strip() == '': return {}
            try:
                if isinstance(data_string, str):
                    data_list = ast.literal_eval(data_string)
                else:
                    data_list = data_string

                if isinstance(data_list, list) and len(data_list) > 0:
                    return data_list[0] if isinstance(data_list[0], dict) else {}
                elif isinstance(data_list, dict):
                    return data_list
                return {}
            except (ValueError, SyntaxError, TypeError):
                return {}

        if 'BuzzBoard Data' in df.columns:
            df['BuzzBoard Data Parsed'] = df['BuzzBoard Data'].apply(safe_parse_buzzboard)
        else:
            df['BuzzBoard Data Parsed'] = [{}] * len(df)

        # Final cleanup
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df.reset_index(drop=True, inplace=True)
        df.fillna('Not Found', inplace=True)

        print(f"Data processing complete. Total prospects: {len(df)}")
        return df

    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        return pd.DataFrame()