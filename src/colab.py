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


def _run(args, token="", quiet=False):
    out = subprocess.run(args, cwd=config.ROOT, capture_output=True, text=True)
    text = (out.stdout + out.stderr).strip()
    if token:
        text = text.replace(token, "***")
    if text and not quiet:
        print(text)
    return out.returncode


def _token_and_url():
    from google.colab import userdata  # only available inside Colab

    owner = config.REPO_SLUG.split("/")[0]
    token = _secret(userdata, "GITHUB_TOKEN")
    _run(["git", "config", "user.name", _secret(userdata, "GIT_NAME", owner)])
    _run(["git", "config", "user.email",
          _secret(userdata, "GIT_EMAIL", f"{owner}@users.noreply.github.com")])
    return token, f"https://{token}@github.com/{config.REPO_SLUG}.git"


def check_ready():
    """Fail before a long download if pushing to GitHub would not work."""
    token, url = _token_and_url()
    if _run(["git", "push", "--dry-run", url, "HEAD:main"], token=token, quiet=True) != 0:
        raise RuntimeError("GITHUB_TOKEN cannot push to the repository (expired, or no 'repo' "
                           "write scope). Create a new token and update the Colab secret.")
    print("GitHub push check passed: downloaded data will be saved during collection.")


def push(message, paths=("data/raw", "figures", "results")):
    """Commit `paths` and push to main. Returns True if the data is on GitHub."""
    token, url = _token_and_url()
    _run(["git", "pull", "-q", "--rebase", "--autostash", "origin", "main"])
    _run(["git", "add", *paths])
    if _run(["git", "diff", "--cached", "--quiet"]) == 0:
        print("Nothing new to commit.")
        return True
    _run(["git", "commit", "-q", "-m", message])
    ok = _run(["git", "push", "-q", url, "HEAD:main"], token=token) == 0
    print(f"Pushed: {message}" if ok else "Push FAILED — check the GITHUB_TOKEN secret.")
    return ok


def checkpoint(message, paths=("data/raw",)):
    """push() for use during long downloads: never interrupts the download.

    A failed push only prints a warning; the files stay on disk and the next
    checkpoint retries them.
    """
    try:
        return push(message, paths)
    except Exception as e:      # network hiccup, git conflict, ...
        print(f"WARNING: checkpoint push failed ({e}); will retry at the next checkpoint.")
        return False
