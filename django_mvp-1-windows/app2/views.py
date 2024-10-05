import os
import pandas as pd
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import sqlite3
import google.generativeai as genai
from dotenv import load_dotenv
import json


def get_gemini_response(question, prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content([prompt[0], question])
    return response.text


def read_sql_query(sql, db):
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    conn.commit()
    conn.close()
    for row in rows:
        print(row)
    return rows


prompt = [
    """
    You are an expert in converting English questions to SQL query!
    The SQL database has the name carbonDB and has the following columns - Scope, Category, 
    Type, Year, Jan to Dec monthly values, Total_Value, Total_Emission,Emission factor.\n  
    \nFor example,\nExample 1 - What is the total Scope 1 emissions for 2023?, 
    the SQL command will be something like this select sum(Total_Emission) from carbonDB where Year='2023' and scope='Scope1';
    \nExample 2 - What is the total Scope 2 emissions for 2022?, 
    the SQL command will be something like this select sum(Total_Emission) from carbonDB where Year='2022' and scope='Scope2';
    \nExample 3 - what is the average emission?, 
    the SQL command will be something like this select sum(Total_Emission)/3 from carbonDB where year in ('2021','2022','2023')
    and scope in ('Scope1','Scope2','Scope3') group by scope; 
    \nExample 4 - What are the top emission categories for Scope 3?, 
    the SQL command will be something like this select category, sum(Total_Emission), Data_Source from carbonDB where year in ('2023') and scope in ('Scope3')
    group by category, sum(Total_Emission) order by sum(Total_Emission) desc; 
    \nExample 5 - Name the top 5 suppliers?, 
    the SQL command will be something like this select distinct Data_Source from carbonDB where scope in ('Scope3');
    also the sql code should not have ``` in beginning or end and sql word in output
    """
]


@csrf_exempt
def ask_query(request):
    if request.method == "POST" and "query" in request.POST and "file" in request.FILES:
        query = request.POST["query"]
        file = request.FILES["file"]
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)

        if file.name.endswith(".xlsx"):
            df = pd.read_excel(file, sheet_name="Raw Data")

        connection = sqlite3.connect("carbonDB")
        df.to_sql("carbonDB", connection, if_exists="replace")
        cursor = connection.cursor()
        print("The isnerted records are")
        data = cursor.execute("""Select * from carbonDB limit 5""")
        for row in data:
            print(row)

        connection.commit()
        connection.close()

        load_dotenv()
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        response = get_gemini_response(query, prompt)
        print(response)
        response = read_sql_query(response, "carbonDB")
        for row in response:
            print(row)
    return JsonResponse({"error": "Invalid request"}, status=400)


@csrf_exempt
def file_upload(request):
    if request.method == "POST" and "file_upload" in request.FILES:
        file_upload = request.FILES["file_upload"]

        try:
            if file_upload.name.endswith(".csv"):
                df = pd.read_csv(file_upload)

            if file_upload.name.endswith(".xlsx"):
                df = pd.read_excel(file_upload, sheet_name="Raw Data")

            null_rows = df[df.isnull().any(axis=1)]
            null_row_ids = null_rows.index.tolist()
            print(null_row_ids)

            duplicated_rows = df[df.duplicated()]
            duplicated_row_ids = duplicated_rows.index.tolist()
            print(duplicated_row_ids)

            df1 = df.iloc[:, 6:18]
            print(df.head())

            outlier_dict = {}

            for index, row in df1.iterrows():
                # Calculate Q1 (25th percentile), Q3 (75th percentile), and IQR for the row
                Q1 = np.percentile(row, 25)
                Q3 = np.percentile(row, 75)
                IQR = Q3 - Q1

                # Define outlier bounds for the row
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # List to store the column indices of outliers
                outlier_indices = []

                # Traverse each value in the row
                for col_index, value in enumerate(row):
                    if value == np.nan:
                        continue
                    # Check if the value is an outlier
                    if value < lower_bound or value > upper_bound:
                        outlier_indices.append(col_index)

                # If there are outliers in this row, store them in the dictionary
                if outlier_indices:
                    outlier_dict[index] = outlier_indices

            print(outlier_dict)
            return JsonResponse(
                {
                    "message": "File processed successfully",
                    "array_2d": df.to_numpy().tolist(),
                    "gaps": null_row_ids,
                    "duplicates": duplicated_row_ids,
                    "anomalies": outlier_dict,
                },
            )

        except Exception as e:
            return JsonResponse(
                {"error": f"Error processing file: {str(e)}"}, status=500
            )

    return JsonResponse({"error": "Invalid request"}, status=400)


# REPLACE GAPS WITH MEAN
def replace_gaps_with_mean(row_index, dataframe):
    # Select the row by index
    row = dataframe.iloc[row_index]

    # Calculate the mean of the row, excluding NaN values
    mean_value = row.mean()

    # Replace NaN values in the row with the mean value
    dataframe.iloc[row_index] = row.fillna(mean_value)

    return dataframe


# DROP ROWS
def drop_rows_by_index(row_index, dataframe):
    # Drop the specified row(s) using the index
    dataframe = dataframe.drop(row_index)

    # Return the updated dataframe
    return dataframe


# APPLY TO BOTH GAPS & ANOMALY TREATMENT
def replace_with_placeholder(placeholder_dict, dataframe):
    # Iterate over each row index in the placeholder_dict
    for row_index, col_val_dict in placeholder_dict.items():
        # For each row, iterate over the columns and their corresponding replacement values
        for col_index, new_value in col_val_dict.items():
            # Replace the value at (row_index, col_index) with new_value
            dataframe.at[row_index, col_index] = new_value

    # Return the modified dataframe
    return dataframe


# REPLACE OUTLIERS WITH MEAN
def replace_outliers_with_mean(outlier_dict_mean, dataframe):
    for row_index, col_indices in outlier_dict_mean.items():
        # Get the row at the given index
        row = dataframe.iloc[row_index]

        # Exclude the outlier columns from the row to calculate the mean
        non_outlier_values = row.drop(labels=col_indices)

        # Calculate the mean of the non-outlier values
        mean_value = non_outlier_values.mean()

        # Replace outlier values with the mean value in the specified columns
        for col_index in col_indices:
            dataframe.iat[row_index, dataframe.columns.get_loc(col_index)] = mean_value

    return dataframe


@csrf_exempt
def handle_cleanups(request):
    if request.method == "POST":
        data = json.loads(request.body)
        gaps_to_mean = request.POST.get("gaps_to_mean", "[]")
        gaps_to_drop = request.POST.get("gaps_to_mean", "[]")
        gaps_to_placeholder_dict = data.get("gaps_to_placeholder_dict", {})
        rows_to_drop = request.POST.get("rows_to_drop", [])
        outlier_dict_placeholder = data.get("outlier_dict_placeholder", {})
        outlier_dict_mean = data.get("outlier_dict_mean", {})
        array_2d = data.get("array_2d", [])

        df = pd.DataFrame(array_2d)
        df = replace_gaps_with_mean(gaps_to_mean, df)
        df = drop_rows_by_index(gaps_to_drop, df)
        df = replace_with_placeholder(gaps_to_placeholder_dict, df)
        df = drop_rows_by_index(rows_to_drop, df)
        df = replace_with_placeholder(outlier_dict_placeholder, df)
        df = replace_outliers_with_mean(outlier_dict_mean, df)

        connection = sqlite3.connect("GHG_DATA")
        df.to_sql("GHG_DATA", connection, if_exists="replace")
        connection.commit()
        connection.close()

        connection = sqlite3.connect("GHG_DATA")

        # Fetch the dataframe
        df = pd.read_sql_query("SELECT * FROM GHG_DATA", connection)

        # Convert dataframe to JSON
        json_data = df.to_json(orient="records")

        # Close the connection
        connection.close()

        # Send JSON response
        return JsonResponse(json_data, safe=False)
    return JsonResponse({"error": "Invalid request"}, status=400)
