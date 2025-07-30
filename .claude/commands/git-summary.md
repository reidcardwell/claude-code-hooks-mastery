# Git Workflow with Summary Display

Run complete git workflow and display detailed results to the user.

## Steps

1. **Execute Git Workflow**: Run git-flow-automator agent to handle staging, commit, and push
2. **Extract Key Information**: Parse the agent response for important details
3. **Display Results**: Show commit hash, message, files changed, and push status to user

## Usage

```
/git-summary
```

## Implementation

Execute the following sequence:

### Step 1: Git Operations
Use the Task tool to run git-flow-automator:
```
Task(description="Execute git workflow", prompt="Update git", subagent_type="git-flow-automator")
```

### Step 2: Information Extraction and Display
After the git-flow-automator completes, extract and display the following information:

**Required Information to Display:**
- Files that were staged and committed
- Commit hash (short form)
- Full commit message
- Push status and target branch
- Any warnings or important notes

**Display Format:**
```
## Git Workflow Complete

**Files Changed**: [list of modified files]
**Commit Hash**: `[short hash]`
**Commit Message**: "[full commit message]"
**Push Status**: [success/failure with branch info]
**Additional Notes**: [any warnings or important info]
```

## Notes

- This command executes git workflow and displays detailed results
- Information is extracted from git-flow-automator agent response
- Provides full visibility into git operations