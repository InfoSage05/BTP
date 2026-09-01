# Project instructions

## Git workflow
- Do not create new git branches or worktrees on your own initiative. Work
  directly on the current branch (`main`) unless the user explicitly asks
  for a branch/worktree. This includes worktrees created implicitly by
  subagents/background isolation (e.g. `D:/BTP.worktrees/...`,
  `.claude/worktrees/...`) -- avoid spawning agents/isolation modes that
  create these for this repo. If one appears anyway, delete it (folder,
  git worktree registration, and branch) as soon as it's safe to do so.
- Do not push commits (or delete branches) without the user's explicit
  consent for that specific push. Committing locally is fine; ask before
  running `git push`.
