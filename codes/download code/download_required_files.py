folders_needed_ids = ["19t2Rrr8w6myhxkyMgo1Gx5pzJvG-lJb4",
                      "1EIxaawaImXejr6BXG7obeLVPsU2YYlqk",
                      "1dWgATmTdr1wQ6JFHrLjixBMh_DvurJEP",
                      "15X4iYM2G659-LG6Gun8UDkDjuUvfRy5h",
                      "18kdbI6e4Bv_tDF5weAPTP9MZzuLGkfbY",
                      "1J_T0df8w2QXOrR1pROapS7WXFFv02FnD",
                      "1IzS9RWjvsYd3aDJB7RfusIr-iedq0KQZ",
                      "1pgJGfUnmJsHl44aG1qF5wkJNNlCCQwFL"
                      ]

# https://github.com/korakot/kora/blob/master/kora/drive.py
# https://raw.githubusercontent.com/segnolin/google-drive-folder-downloader/master/download.py
def download_folder_from_drive(folder_id):
    """ kora libarary """
    # authenticate
    from google.colab import auth
    auth.authenticate_user()
    # get folder_name
    from googleapiclient.discovery import build
    service = build('drive', 'v3')
    folder_name = service.files().get(fileId=folder_id).execute()['name']
    # install library 
    # url = 'https://github.com/segnolin/google-drive-folder-downloader/raw/master/download.py'
    # path = '/usr/local/lib/python3.6/dist-packages/download.py'
    # urlretrieve(url, path)
    # recursive download
    from download_lib import download_folder
    download_folder(service, folder_id, './', folder_name)
    return folder_name

def download_required_folders():
  for i in folders_needed_ids:
    download_folder_from_drive(i)  
  