import os
import pickle

"""
    These functions are useful in saving the record of uploaded images
    - When an image is uplaoded for first time, a record is stored in the form of a pickle file
    - If the image is uploaded again, the record is fetched from the pickle file thus saving time
"""


def store_object(obj):
    ''' 
    Creates a pickle file for the object
    '''
    os.makedirs("./utils/pickle",
                exist_ok=True)  # Create the pickle folder if it doesn't exist
    with open(f"./utils/pickle/{os.path.splitext(obj.display_name)[0]}.pkl", 'wb') as file:
        pickle.dump(obj, file)


def fetch_object(file_path):
    '''
    Fetches the object from the pickle file
    '''
    file_name = os.path.basename(file_path)  # Get the file name from the path
    file_name_without_extension = os.path.splitext(file_name)[0]
    try:

        with open(f'./utils/pickle/{file_name_without_extension}.pkl', 'rb') as file:
            obj = pickle.load(file)
            return obj

    except Exception as e:
        # print(e)
        return None
