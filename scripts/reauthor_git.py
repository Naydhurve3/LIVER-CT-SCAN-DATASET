import subprocess

# Re-author command
cmd = [
    "git", "rebase", "-i", "--root",
    "--exec", "git commit --amend --author=\"NAYAN DHURVE <122363521+Naydhurve3@users.noreply.github.com>\" --no-edit"
]

env = dict(os.environ) if 'os' in globals() else None
import os
env = dict(os.environ)
env["GIT_EDITOR"] = "true"  # Auto-save rebase todo list

print("Executing rebase...")
res = subprocess.run(cmd, env=env, capture_output=True, text=True)
print("STDOUT:", res.stdout)
print("STDERR:", res.stderr)
print("Return code:", res.returncode)
