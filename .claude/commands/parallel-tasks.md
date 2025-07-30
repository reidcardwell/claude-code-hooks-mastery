# Parallel Task Execution

Execute multiple Taskmaster tasks in parallel with intelligent dependency resolution, contextual implementation, and automated git workflow.

## Usage

```
/parallel-tasks 1.2,1.3,1.4 [--dry-run]
```

## Arguments

- **Task IDs**: Comma-separated list of Taskmaster task IDs (e.g., `1.2,1.3,2.1`)
- **--dry-run**: Show execution plan without executing (optional, place at end)

## Implementation

Execute the following comprehensive workflow:

### Phase 1: Planning & Validation
1. **Parse Task IDs**: Validate format and extract task identifiers
2. **Fetch Task Details**: Use Taskmaster to get full task information including descriptions, dependencies, and current status
3. **Dependency Resolution**: 
   - Build complete dependency graph
   - Auto-include any unmet dependencies
   - Notify user of auto-included tasks: "Auto-including dependencies: 1.1, 2.3"
   - Detect circular dependencies and abort if found
4. **Execution Planning**:
   - Group tasks into parallel execution batches (max 4 simultaneous)
   - Respect dependency order within batches
   - Create execution timeline
5. **Dry-run Check**: If `--dry-run` flag present, display execution plan and exit

### Phase 2: Parallel Task Execution
Execute tasks in dependency-respecting parallel batches with progress notifications:

**For each task in batch:**
```
Task(description="Execute Taskmaster task [ID]", 
     prompt="Execute Taskmaster task [ID]: [TASK_TITLE]. Use hybrid contextual approach: 1) Read task details from Taskmaster, 2) Analyze relevant codebase patterns and architecture, 3) Implement following existing conventions, 4) Auto-detect and run tests if needed. Focus on quality implementation that integrates well with existing code.", 
     subagent_type="general-purpose")
```

**Success Criteria per Task:**
- Implementation completed (code written/modified)
- Auto-detected tests pass (if tests exist or were created)
- No syntax/compilation errors

**Progress Notifications:**
Display real-time status as tasks complete:
- "✅ Task 1.2 completed: Implemented user authentication system"
- "❌ Task 1.3 failed: Syntax error in validation logic"
- "⏭️ Task 2.1 skipped: Depends on failed task 1.3"

### Phase 3: Error Handling & Dependency Impact
**If a task fails:**
1. Mark task as failed
2. Identify all downstream dependent tasks in current batch
3. Skip dependent tasks and mark as "Skipped due to failed dependency [ID]"
4. Continue with independent tasks
5. Alert user: "Task [ID] failed. Skipping dependent tasks: [list]"

### Phase 4: Single Git Commit
**After all parallel execution completes, commit all changes together:**
```
Task(description="Commit all parallel task changes", 
     prompt="Update git with comprehensive commit for parallel task execution:

Complete parallel tasks: [X successful, Y failed, Z skipped]

✅ Task [ID]: [implementation summary]
✅ Task [ID]: [implementation summary]  
❌ Task [ID]: [failure reason]
⏭️ Task [ID]: SKIPPED (depends on failed task [ID])

Tests: [X passed, Y failed]
Auto-included dependencies: [list]
Files modified: [summary]", 
     subagent_type="git-flow-automator")
```

**Commit Strategy**: Include all work (successful and failed attempts) with clear notation for debugging and transparency.

### Phase 5: Batch Status Update
**After git commit completes:**
```
Task(description="Update Taskmaster status", 
     prompt="Mark the following tasks as complete in Taskmaster: [successful_task_ids]. Tasks completed successfully with implementations and single batch commit.", 
     subagent_type="taskmaster-task-updater")
```

### Phase 6: Comprehensive Session Update
**Final step:**
```
Task(description="Update development session", 
     prompt="Update session with detailed parallel task execution results: 
     - Executed tasks: [all_attempted_tasks] 
     - Successful completions: [successful_tasks] 
     - Failed tasks: [failed_tasks] 
     - Skipped due to dependencies: [skipped_tasks]
     - Key implementations: [summary of what was built]
     - Single comprehensive git commit created
     - Auto-detected tests run: [test_results]", 
     subagent_type="session-manager")
```

## Execution Strategy

### Parallel Batching Logic
1. **Wave 1**: Execute all tasks with no dependencies (up to 4 parallel)
2. **Wave 2**: Execute tasks whose dependencies completed in Wave 1 (up to 4 parallel)  
3. **Wave N**: Continue until all tasks processed or blocked by failures

### Contextual Implementation Approach
Each general-purpose agent follows the **Hybrid Contextual Strategy**:
- **Project Context**: Analyze existing codebase architecture and patterns
- **Task Focus**: Implement specific task requirements
- **Quality Integration**: Ensure new code follows established conventions
- **Test Awareness**: Auto-detect testing needs and execute when appropriate

### Commit Message Format
Single comprehensive commit with detailed breakdown:
```
Complete parallel tasks: [X successful, Y failed, Z skipped]

✅ Task [ID]: [implementation summary]
✅ Task [ID]: [implementation summary]  
❌ Task [ID]: [failure reason]
⏭️ Task [ID]: SKIPPED (depends on failed task [ID])

Tests: [X passed, Y failed]
Auto-included dependencies: [list]
Files modified: [list of files]

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

## Example Usage

```bash
# Execute three tasks in parallel with dependency resolution
/parallel-tasks 2.1,2.3,3.1

# Preview execution plan without running
/parallel-tasks 1.2,1.4,2.2 --dry-run
```

## Expected Output Summary

```
## Parallel Task Execution Complete

**Tasks Processed**: 4 (including 1 auto-included dependency)
**Successful**: 3 tasks completed with commits
**Failed**: 1 task (dependency issues)
**Skipped**: 0 tasks

**Execution Timeline**:
Wave 1: Tasks 1.1, 2.1 (parallel)
Wave 2: Tasks 1.2, 2.3 (parallel, after 1.1 completed)

**Git Commit**: 1 comprehensive commit with detailed breakdown
**Session Updated**: Comprehensive progress logged
```

## Error Scenarios

- **Invalid Task IDs**: Abort with clear error message
- **Circular Dependencies**: Abort with dependency chain explanation  
- **Taskmaster Unavailable**: Abort with Taskmaster connection guidance
- **Git Conflicts**: Handle per-task, continue with others
- **Implementation Failures**: Skip dependents, continue with independent tasks

## Progress Notification System

During execution, the command provides real-time feedback:

```
🚀 Starting parallel task execution...
📋 Auto-including dependencies: 1.1
⚡ Wave 1: Executing tasks 1.1, 2.1 (parallel)
✅ Task 1.1 completed: Set up authentication middleware
✅ Task 2.1 completed: Created user profile API
⚡ Wave 2: Executing tasks 1.2, 2.3 (parallel) 
✅ Task 1.2 completed: Added login/logout endpoints
❌ Task 2.3 failed: Database connection error
⏭️ Task 3.1 skipped: Depends on failed task 2.3
🔄 Committing all changes...
✅ Git commit successful: 1d4f2a8
📝 Updating Taskmaster status...
📊 Session updated with comprehensive results
```

## Notes

- Maximum 4 simultaneous task executions for performance
- True parallelism within dependency constraints
- Comprehensive error handling with downstream impact analysis
- Single comprehensive git commit includes all work (successful and failed)
- Real-time progress notifications keep user informed
- Session logging includes detailed breakdown of all activities
- Failed work preserved in commit for debugging purposes