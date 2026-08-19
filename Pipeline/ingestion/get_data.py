import os
import io
import pickle

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request


# ============================================================
# CONFIGURATION
# ============================================================

SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly"
]

CREDENTIALS_FILE = "credentials/credentials.json"

TOKEN_FILE = "token.pickle"

RAW_FOLDER_ID = "12eN15LBIXqZBxJ0ss4eijdNLtzawTAml"

OUTPUT_FOLDER = "local/raw"


# ============================================================
# GOOGLE DRIVE AUTHENTICATION
# ============================================================

def authenticate_google_drive():

    creds = None

    # Check if we already have a saved token
    if os.path.exists(TOKEN_FILE):

        with open(TOKEN_FILE, "rb") as token:
            creds = pickle.load(token)

    # If credentials don't exist or aren't valid
    if not creds or not creds.valid:

        # Refresh expired credentials
        if creds and creds.expired and creds.refresh_token:

            creds.refresh(Request())

        else:

            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE,
                SCOPES
            )

            creds = flow.run_local_server(port=0)

        # Save credentials for next time
        with open(TOKEN_FILE, "wb") as token:

            pickle.dump(creds, token)

    # Create Google Drive service
    service = build(
        "drive",
        "v3",
        credentials=creds
    )

    return service


# ============================================================
# GET FILES FROM RAW FOLDER
# ============================================================

def get_raw_files(service):

    query = (
        f"'{RAW_FOLDER_ID}' in parents "
        f"and trashed = false"
    )

    results = service.files().list(
        q=query,
        fields="files(id, name, mimeType, size, modifiedTime)",
        pageSize=1000
    ).execute()

    files = results.get("files", [])

    return files


# ============================================================
# DOWNLOAD ONE FILE
# ============================================================

def download_file(
    service,
    file_id,
    file_name,
    output_folder
):

    # Create output folder if it doesn't exist
    os.makedirs(
        output_folder,
        exist_ok=True
    )

    file_path = os.path.join(
        output_folder,
        file_name
    )

    # Skip file if already downloaded
    if os.path.exists(file_path):

        print(
            f"[SKIP] Already exists: {file_name}"
        )

        return file_path

    print(
        f"[DOWNLOAD] {file_name}"
    )

    request = service.files().get_media(
        fileId=file_id
    )

    with open(file_path, "wb") as file:

        downloader = MediaIoBaseDownload(
            file,
            request
        )

        done = False

        while not done:

            status, done = downloader.next_chunk()

            if status:

                progress = int(
                    status.progress() * 100
                )

                print(
                    f"    Progress: {progress}%"
                )

    print(
        f"[DONE] {file_name}"
    )

    return file_path


# ============================================================
# INGESTION PIPELINE
# ============================================================

def ingest_data():

    print("\n==============================")
    print(" Google Drive Ingestion")
    print("==============================\n")

    # 1. Authenticate
    service = authenticate_google_drive()

    print("[OK] Google Drive authenticated.\n")

    # 2. Get files
    files = get_raw_files(service)

    print(
        f"[INFO] Found {len(files)} files in RAW folder.\n"
    )

    # 3. Download files
    downloaded_files = []

    for file in files:
        file_id = file["id"]
        file_name = file["name"]
        mime_type = file["mimeType"]

        print(
            f"Found: {file_name} | {mime_type}"
        )

        # Download only PDF files
        if mime_type == "application/pdf":

            path = download_file(
                service=service,
                file_id=file_id,
                file_name=file_name,
                output_folder=OUTPUT_FOLDER
            )

            downloaded_files.append(path)

        else:

            print(
                f"[SKIP] Not a PDF: {file_name}"
            )

    print("\n==============================")
    print(" Ingestion Completed")
    print("==============================")

    print(
        f"\nDownloaded files: "
        f"{len(downloaded_files)}"
    )

    return downloaded_files


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    ingest_data()