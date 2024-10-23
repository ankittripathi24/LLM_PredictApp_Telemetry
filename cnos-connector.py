# from cnosdb_connector import connect
# import pandas as pd

# conn = connect(url="http://127.0.0.1:8902/", user="root", password="")
# resp = conn.execute("SHOW DATABASES")
# print("SHOW DATABASES: ", resp)


# # conn.create_database("air")
# resp = conn.list_database()
# print("LIST DATABASES: ", resp)

# # resp = pd.read_sql("SHOW DATABASES", conn)
# # print(resp)


# import os

# query = "CREATE TABLE air (  visibility DOUBLE, temperature DOUBLE, pressure DOUBLE, TAGS(station));"

# # conn.execute(query)
# resp = conn.execute(query)
# print(resp)



# resp = conn.execute("INSERT INTO air (TIME, station, visibility, temperature, pressure) VALUES (1666165202490401000, 'XiaoMaiDao', 56, 69, 77);")
# print(resp)


# resp = conn.execute("SELECT * FROM air;")
# print(resp)


import os
import pandas as pd
from cnosdb_connector import connect

def insert_data_to_tpusr12():
    # Connect to CNOSDB
    conn = connect(url="http://127.0.0.1:8902/", user="root", password="")
    
    # Directory containing the CSV files
    csv_directory = r"D:\tenant_TimeSeriesdata_tpusr12\latest data"  # Use raw string to avoid escaping backslashes
    
    # List to hold the data to be inserted
    data_to_insert = []

    # Read all CSV files in the directory
    for filename in os.listdir(csv_directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(csv_directory, filename)
            df = pd.read_csv(file_path, delimiter=';')
            
            # Extract the required columns
            if '_time' in df.columns and 'IntTriangle1000ms' in df.columns:
                for index, row in df.iterrows():
                    data_to_insert.append((row['_time'], row['IntTriangle1000ms']))

    # Create the table if it doesn't exist
    create_table_query = """
    CREATE TABLE IF NOT EXISTS tpusr12 (
        
        IntTriangle1000ms DOUBLE
    );
    """
    conn.execute(create_table_query)

    # Insert data into the table
    if data_to_insert:
        insert_query = "INSERT INTO tpusr12 (TIME, IntTriangle1000ms) VALUES "
        values = ", ".join([f"('{time}', {value})" for time, value in data_to_insert])
        insert_query += values + ";"
        conn.execute(insert_query)
        print(f"Inserted {len(data_to_insert)} rows into tpusr12.")
    else:
        print("No data to insert.")


def read_data_from_tpusr12():
    """
    Function to read data from the tpusr12 table.
    
    Returns:
        pd.DataFrame: DataFrame containing the data from the tpusr12 table.
    """
    # Connect to CNOSDB
    conn = connect(url="http://127.0.0.1:8902/", user="root", password="")
    
    # Query to select data from the tpusr12 table
    select_query = "SELECT * FROM tpusr12;"
    
    # Execute the query and fetch the results
    # result = conn.execute(select_query)
    
    # print(result)

    resp = pd.read_sql(select_query, conn)
    print(resp)

    
    
    return resp

# Example usage
if __name__ == "__main__":
    # insert_data_to_tpusr12()
    df = read_data_from_tpusr12()
    # print(df)