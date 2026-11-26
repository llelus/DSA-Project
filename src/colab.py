"""Colab helper: commit and push generated files back to GitHub.

Credentials and identity come from Colab secrets (key icon in the sidebar):
  GITHUB_TOKEN  personal access token with repo scope (required)
  GIT_NAME      commit author name   (optional; default: the repo owner)
  GIT_EMAIL     commit author e-mail (optional; default: GitHub no-reply address)
so nothing personal is hard-coded in the notebooks.
"""
import subprocess

from . import config


def _secret(userdata, name, default=None):
    try:
        value = userdata.get(name)
    except Exception:            # SecretNotFoundError / NotebookAccessError
        value = None
    if not value and default is None:
        raise RuntimeError(f"Colab secret {name} is missing (key icon in the left sidebar).")
    return value or default


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

    owner = config.REPO_SLUG.split("/")[0]
    token = _secret(userdata, "GITHUB_TOKEN")
    _run(["git", "config", "user.name", _secret(userdata, "GIT_NAME", owner)])
    _run(["git", "config", "user.email",
          _secret(userdata, "GIT_EMAIL", f"{owner}@users.noreply.github.com")])
    _run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    _run(["git", "add", *paths])
    if _run(["git", "diff", "--cached", "--quiet"]) == 0:
        print("Nothing to commit.")
        return
    _run(["git", "commit", "-m", message])
    url = f"https://{token}@github.com/{config.REPO_SLUG}.git"
    code = _run(["git", "push", url, "HEAD:main"], token=token)
    print("Pushed." if code == 0 else "Push FAILED — check the GITHUB_TOKEN secret.")
