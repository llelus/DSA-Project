"""Colab helper: commit and push generated files back to GitHub.

Credentials and identity come from Colab secrets (key icon in the sidebar):
  GITHUB_TOKEN  personal access token with repo scope
  GIT_NAME      commit author name
  GIT_EMAIL     commit author e-mail
so nothing personal is hard-coded in the notebooks.
"""
import subprocess

from . import config


def _run(args, token=""):
    out = subprocess.run(args, cwd=config.ROOT, capture_output=True, text=True)
    text = (out.stdout + out.stderr).strip()
    if token:
        text = text.replace(token, "***")
    if text:
        print(text)
    return out.returncode


def push(message, paths=("data/raw", "figures", "results")):
    from google.colab import userdata  # only available inside Colab

    token = userdata.get("GITHUB_TOKEN")
    _run(["git", "config", "user.name", userdata.get("GIT_NAME")])
    _run(["git", "config", "user.email", userdata.get("GIT_EMAIL")])
    _run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    _run(["git", "add", *paths])
    if _run(["git", "diff", "--cached", "--quiet"]) == 0:
        print("Nothing to commit.")
        return
    _run(["git", "commit", "-m", message])
    url = f"https://{token}@github.com/{config.REPO_SLUG}.git"
    code = _run(["git", "push", url, "HEAD:main"], token=token)
    print("Pushed." if code == 0 else "Push FAILED — check the GITHUB_TOKEN secret.")
