from huggingface_hub import HfApi, create_repo
api = HfApi()


repo_id = "rss-hiverobots/grt_pick_multiple_toys_21jan_1cam_prompt2_id"

# Create the repo if it doesn't exist
create_repo(repo_id, repo_type="dataset", exist_ok=True)

# Upload a single file
# api.upload_file(
#     path_or_fileobj="path/to/local/data.csv",
#     path_in_repo="data.csv",
#     repo_id="username/dataset-name",
#     repo_type="dataset",
# )

# Upload an entire folder
api.upload_folder(
    folder_path="/home/rss/.cache/huggingface/lerobot/rss-hiverobots/grt_pick_multiple_toys_21jan_1cam_prompt2_id",
    repo_id=repo_id,
    repo_type="dataset",
)