# import requests

# # Server URL for prediction endpoint
# url = "http://127.0.0.1:8000/predict/"  # Update this with your FastAPI server's actual URL

# # Path to the file you want to upload
# file_path = "C:/example_input_images/ceT1/vs_gk_0000.nii.gz"

# try:
#     # Open the file in binary mode
#     with open(file_path, "rb") as file:
#         # Prepare the file to send with a proper filename and content type
#         files = {"file": ("vs_gk_0000.nii.gz", file, "application/gzip")}

#         # Send a POST request with the file
#         response = requests.post(url, files=files)

#     # Check the response status code
#     if response.status_code == 200:
#         # If successful, the server should return the .nii.gz file or a success message
#         print("Status Code:", response.status_code)
#         print("Response Content-Type:", response.headers.get("Content-Type"))

#         # If the response is a file, save it locally
#         if response.headers.get("Content-Type") == "application/gzip":
#             output_file_path = "C:/example_output_images/predicted_vs_gk.nii.gz"  # Output path
#             with open(output_file_path, "wb") as output_file:
#                 output_file.write(response.content)
#             print(f"File saved to {output_file_path}")
#         else:
#             # If the response is not a file, print the JSON response
#             print("Response Body:", response.json())
#     else:
#         print("Error occurred!")
#         print("Status Code:", response.status_code)
#         print("Response Body:", response.text)

# except FileNotFoundError:
#     print(f"File not found: {file_path}. Please check the file path and try again.")
# except requests.exceptions.RequestException as e:
#     print(f"An error occurred while making the request: {e}")
# except Exception as e:
#     print(f"An unexpected error occurred: {e}")




import requests

url = "http://127.0.0.1:8000/predict/"  # Update with your API URL
file_path = "C:/example_input_images/ceT1/vs_gk_0000.nii.gz"  # File to upload
output_path = "downloaded_file.nii.gz"  # Path to save the returned file

with open(file_path, "rb") as file:
    # Send the file to the API
    response = requests.post(url, files={"file": file})

    # Check if the request was successful
    if response.status_code == 200:
        # Save the returned file
        with open(output_path, "wb") as output_file:
            output_file.write(response.content)
        print(f"File saved as {output_path}")
    else:
        print(f"Failed to fetch the file. Status code: {response.status_code}")
        print(response.json())  # If the API sends error details
