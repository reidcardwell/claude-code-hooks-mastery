# Create Pull Request

Create an intelligent pull request with automatic parent branch detection and comprehensive description generation.

## Usage

```
/create-pr [--draft] [--base branch-name] [--title "Custom Title"]
```

## Arguments

- **--draft**: Create as draft pull request (optional)
- **--base branch-name**: Override automatic parent detection with specific base branch (optional)
- **--title "Custom Title"**: Use custom PR title instead of auto-generated (optional)

## Implementation

Execute intelligent pull request creation workflow:

```
Task(description="Create intelligent pull request", 
     prompt="Create pull request with intelligent parent branch detection and comprehensive description:
     
     1. Analyze current branch and detect most appropriate parent branch using merge-base
     2. Generate descriptive PR title based on branch name and recent commits
     3. Create comprehensive PR description including:
        - Summary of changes made
        - File changes breakdown
        - Commit history since branch point
        - Auto-detected test information
        - Proper formatting with markdown
     4. Create PR targeting detected parent branch
     5. Provide PR URL and next steps
     
     Handle flags: $ARGUMENTS
     - Parse --draft flag for draft PR creation
     - Parse --base for manual parent override  
     - Parse --title for custom title override
     
     Use intelligent merge-base analysis to find parent:
     - Check main, dev, develop, master branches
     - Select branch with most recent common ancestor
     - Validate target branch exists on remote
     
     Generate professional PR description with task breakdown if applicable.", 
     subagent_type="pr-creator")
```

## Example Usage

```bash
# Create PR with auto-detected parent and title
/create-pr

# Create draft PR
/create-pr --draft

# Override parent branch detection
/create-pr --base dev

# Custom title and draft mode
/create-pr --draft --title "WIP: User authentication system"

# All options combined
/create-pr --draft --base main --title "Feature: Complete user management system"
```

## Expected Workflow

1. **Branch Analysis**: Detect current feature branch and validate state
2. **Parent Detection**: Use merge-base to find most appropriate target branch
3. **Content Generation**: Create comprehensive PR title and description
4. **PR Creation**: Execute `gh pr create` with proper targeting
5. **Confirmation**: Provide PR URL and next action recommendations

## Integration Notes

- **Works independently** or as part of `/parallel-tasks` workflow
- **Respects project conventions** like CODEOWNERS and PR templates
- **Handles common scenarios** like protected branches and authentication
- **Provides clear feedback** on parent branch selection reasoning

This command bridges the gap between development completion and code review, ensuring PRs are properly targeted and contain comprehensive information for reviewers.