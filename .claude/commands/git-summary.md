# Git Workflow Only

Run git workflow for staging, commit, and push operations.

## Steps

1. **Execute Git Workflow**: Run git-flow-automator agent to handle staging, commit, and push
2. **Manual Summary Option**: User can manually request TTS summary if desired using "tts summary"

## Usage

```
/git-summary
```

## Implementation

Execute git workflow only:

### Git Operations
Use the Task tool to run git-flow-automator:
```
Task(description="Execute git workflow", prompt="Update git", subagent_type="git-flow-automator")
```

## Optional Manual TTS Summary

If you want audio feedback after git operations, manually use:
- Say "tts summary" to activate enhanced-work-summary agent
- Provide context about what was committed
- Agent will generate audio feedback

## Notes

- This command executes git workflow only
- TTS summary is available on manual request
- Prevents automatic loops while maintaining functionality