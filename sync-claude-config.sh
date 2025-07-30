#!/bin/bash

# Claude Config Sync Script
# Synchronizes .claude directories (agents, hooks, commands) from this project to a target directory

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Global variables
DRY_RUN=false
TARGET_DIR=""
BACKUP_DIR=""
APPLY_TO_ALL=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIRS=("agents" "hooks" "commands")

# Sub-agent trigger rules content to inject
SUB_AGENT_RULES='## Sub-Agent Trigger Rules

### Automatic Sub-Agent Activation
When these keywords/phrases are detected, ALWAYS use the specified sub-agent via the Task tool:

**Git Operations:**
- Keywords: "update git", "commit", "push", "git workflow", "stage changes", "create PR", "merge"
- Action: Use Task tool with git-flow-automator agent

**Performance Issues:**
- Keywords: "slow", "performance", "optimize", "bottleneck", "speed up", "improve performance"
- Action: Use Task tool with performance-optimizer agent

**Work Completion:**
- Keywords: "tts summary", "audio summary"
- Action: Use Task tool with enhanced-work-summary agent
- Note: Only when explicitly requested by user

**Complex Search/Analysis:**
- Keywords: "search for", "find in codebase", "multi-step", "analyze system", "comprehensive search"
- Action: Use Task tool with general-purpose agent

**Task Management Updates:**
- Keywords: "update task", "modify task", "enhance task", "expand task", "task hierarchy"
- Action: Use Task tool with taskmaster-task-updater agent

### Sub-Agent Usage Protocol
1. **ALWAYS prioritize sub-agents over direct tool usage** for specialized operations
2. **Before using Bash, Edit, or other direct tools**, check if a sub-agent exists for the task type
3. **Default to Task tool** when the operation matches any trigger keywords above
4. **Sub-agents provide enhanced capabilities** through specialized knowledge and tool coordination
'

# Function to print colored output
print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to show usage
show_help() {
    cat << EOF
Usage: $0 [OPTIONS] TARGET_DIRECTORY

Synchronizes .claude directories (agents, hooks, commands) from this project to TARGET_DIRECTORY.

OPTIONS:
    -h, --help      Show this help message
    -n, --dry-run   Show what would be done without making changes

BEHAVIOR:
    - Missing files: Copied from source to target
    - Newer source files: Overwrite target files (with backup)
    - Older source files: Prompt user for action
    - Ignores OS hidden files (.DS_Store, Thumbs.db, etc.)
    - Creates timestamped backups in target/.claude/backups/

EXAMPLES:
    $0 /path/to/other/project
    $0 --dry-run /path/to/other/project
EOF
}

# Function to check if file should be ignored
should_ignore() {
    local filename="$1"
    case "$filename" in
        .DS_Store|Thumbs.db|desktop.ini|.Trashes|.Spotlight-V100|.fseventsd)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# Function to check if directory should be ignored
should_ignore_directory() {
    local dirpath="$1"
    case "$dirpath" in
        */.claude/hooks/logs|*/.claude/hooks/logs/*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# Function to create basic CLAUDE.md for a project
create_basic_claude_md() {
    local target_file="$1"
    local project_name="$(basename "$(dirname "$target_file")")"
    
    print_color "$GREEN" "Creating basic CLAUDE.md for project: $project_name"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        cat > "$target_file" << EOF
# $project_name - Claude Code Integration Guide

## Project Overview

This file provides guidance to Claude Code when working with code in this repository.

## Workflow Preferences

Before executing any directive:
1. Take a deep breath, you are a 10x senior developer.
2. Read the current todo list using TodoRead to understand pending tasks
3. Use TodoWrite to plan and track complex tasks throughout the conversation
4. Write all code as efficiently and with as few lines as possible.
5. If you're unsure about any aspect or if the request lacks necessary information, say "I don't have enough information to confidently assess this."

$SUB_AGENT_RULES

## Personal Notes

- Prefer concise responses
- Focus on practical implementation over explanations
EOF
        print_color "$GREEN" "Created: $target_file"
    else
        print_color "$BLUE" "[DRY RUN] Would create basic CLAUDE.md: $target_file"
    fi
}

# Function to merge sub-agent rules into existing CLAUDE.md
merge_sub_agent_rules() {
    local target_file="$1"
    local temp_file="${target_file}.tmp"
    
    print_color "$GREEN" "Merging sub-agent rules into existing CLAUDE.md"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        # Check if sub-agent rules already exist
        if grep -q "## Sub-Agent Trigger Rules" "$target_file"; then
            print_color "$YELLOW" "Sub-agent rules section already exists, updating..."
            
            # Remove existing sub-agent rules section and everything until next ## section
            awk '
                /^## Sub-Agent Trigger Rules/ { skip=1; next }
                /^## / && skip { skip=0 }
                !skip { print }
            ' "$target_file" > "$temp_file"
            
            # Add updated sub-agent rules before the first ## section after project overview
            awk -v rules="$SUB_AGENT_RULES" '
                BEGIN { added=0 }
                /^## / && !added && !/^## Project Overview/ && !/^## Sub-Agent Trigger Rules/ {
                    print rules
                    print ""
                    added=1
                }
                { print }
            ' "$temp_file" > "$target_file"
            
            rm "$temp_file"
        else
            print_color "$YELLOW" "Adding sub-agent rules section..."
            
            # Find insertion point (after Project Overview or at beginning)
            awk -v rules="$SUB_AGENT_RULES" '
                BEGIN { added=0 }
                /^## / && !added && !/^## Project Overview/ {
                    print rules
                    print ""
                    added=1
                }
                { print }
                END {
                    if (!added) {
                        print ""
                        print rules
                    }
                }
            ' "$target_file" > "$temp_file"
            
            mv "$temp_file" "$target_file"
        fi
        
        print_color "$GREEN" "Updated: $target_file"
    else
        print_color "$BLUE" "[DRY RUN] Would merge sub-agent rules into: $target_file"
    fi
}

# Function to handle CLAUDE.md file specially
handle_claude_md() {
    local target_file="$TARGET_DIR/CLAUDE.md"
    
    if [[ ! -f "$target_file" ]]; then
        create_basic_claude_md "$target_file"
    else
        merge_sub_agent_rules "$target_file"
    fi
}

# Function to create backup directory structure
create_backup_dir() {
    local timestamp=$(date +"%Y-%m-%d_%H%M%S")
    BACKUP_DIR="${TARGET_DIR}/.claude/backups/${timestamp}"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        mkdir -p "$BACKUP_DIR"
        print_color "$GREEN" "Created backup directory: $BACKUP_DIR"
    else
        print_color "$BLUE" "[DRY RUN] Would create backup directory: $BACKUP_DIR"
    fi
}

# Function to backup a file
backup_file() {
    local target_file="$1"
    local relative_path="${target_file#$TARGET_DIR/.claude/}"
    local backup_file="$BACKUP_DIR/$relative_path"
    local backup_parent_dir="$(dirname "$backup_file")"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        mkdir -p "$backup_parent_dir"
        cp "$target_file" "$backup_file"
        print_color "$YELLOW" "  Backed up to: $backup_file"
    else
        print_color "$BLUE" "  [DRY RUN] Would backup to: $backup_file"
    fi
}

# Function to get file modification time (cross-platform)
get_file_mtime() {
    local file="$1"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        stat -f "%m" "$file"
    else
        # Linux
        stat -c "%Y" "$file"
    fi
}

# Function to format timestamp for display
format_timestamp() {
    local timestamp="$1"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        date -r "$timestamp" '+%Y-%m-%d %H:%M:%S'
    else
        # Linux
        date -d "@$timestamp" '+%Y-%m-%d %H:%M:%S'
    fi
}

# Function to prompt user for action
prompt_user() {
    local source_file="$1"
    local target_file="$2"
    local source_time="$3"
    local target_time="$4"
    
    # If we have a global choice, apply it
    if [[ -n "$APPLY_TO_ALL" ]]; then
        return
    fi
    
    echo
    print_color "$YELLOW" "Source file is older than target:"
    echo "  File: ${source_file#$SCRIPT_DIR/}"
    echo "  Source: $(format_timestamp "$source_time")"
    echo "  Target: $(format_timestamp "$target_time")"
    echo
    
    while true; do
        read -p "Action: (o)verwrite, (s)kip, (d)iff, (a)ll overwrite, (A)ll skip? [s]: " choice
        case "${choice:-s}" in
            o|O)
                APPLY_TO_ALL=""
                return 0
                ;;
            s|S)
                APPLY_TO_ALL=""
                return 1
                ;;
            d|D)
                echo
                print_color "$BLUE" "Differences (source -> target):"
                diff "$source_file" "$target_file" || true
                echo
                ;;
            a)
                APPLY_TO_ALL="overwrite"
                return 0
                ;;
            A)
                APPLY_TO_ALL="skip"
                return 1
                ;;
            *)
                echo "Please choose: (o)verwrite, (s)kip, (d)iff, (a)ll overwrite, (A)ll skip"
                ;;
        esac
    done
}

# Function to handle file based on comparison
handle_file() {
    local source_file="$1"
    local target_file="$2"
    
    if [[ ! -f "$target_file" ]]; then
        # File doesn't exist in target - copy it
        local target_parent_dir="$(dirname "$target_file")"
        print_color "$GREEN" "Copying new file: ${source_file#$SCRIPT_DIR/}"
        
        if [[ "$DRY_RUN" == "false" ]]; then
            mkdir -p "$target_parent_dir"
            cp "$source_file" "$target_file"
        else
            print_color "$BLUE" "[DRY RUN] Would copy to: $target_file"
        fi
    else
        # File exists - compare timestamps
        local source_time=$(get_file_mtime "$source_file")
        local target_time=$(get_file_mtime "$target_file")
        
        if [[ "$source_time" -gt "$target_time" ]]; then
            # Source is newer - overwrite with backup
            print_color "$GREEN" "Updating newer file: ${source_file#$SCRIPT_DIR/}"
            
            if [[ "$DRY_RUN" == "false" ]]; then
                backup_file "$target_file"
                cp "$source_file" "$target_file"
            else
                print_color "$BLUE" "[DRY RUN] Would backup and overwrite: $target_file"
            fi
        elif [[ "$source_time" -lt "$target_time" ]]; then
            # Source is older - prompt user
            local should_overwrite=false
            
            if [[ "$APPLY_TO_ALL" == "overwrite" ]]; then
                should_overwrite=true
            elif [[ "$APPLY_TO_ALL" == "skip" ]]; then
                should_overwrite=false
            elif [[ "$DRY_RUN" == "false" ]]; then
                if prompt_user "$source_file" "$target_file" "$source_time" "$target_time"; then
                    should_overwrite=true
                fi
            fi
            
            if [[ "$should_overwrite" == "true" ]]; then
                print_color "$YELLOW" "Overwriting with older file: ${source_file#$SCRIPT_DIR/}"
                if [[ "$DRY_RUN" == "false" ]]; then
                    backup_file "$target_file"
                    cp "$source_file" "$target_file"
                else
                    print_color "$BLUE" "[DRY RUN] Would backup and overwrite: $target_file"
                fi
            else
                print_color "$YELLOW" "Skipping older file: ${source_file#$SCRIPT_DIR/}"
            fi
        else
            # Files are identical (same timestamp)
            print_color "$BLUE" "Files identical: ${source_file#$SCRIPT_DIR/}"
        fi
    fi
}

# Function to sync a directory recursively
sync_directory() {
    local source_dir="$1"
    local target_dir="$2"
    
    print_color "$GREEN" "Syncing directory: $source_dir"
    
    # Find all files in source directory
    while IFS= read -r -d '' source_file; do
        local filename="$(basename "$source_file")"
        
        # Skip ignored files
        if should_ignore "$filename"; then
            continue
        fi
        
        # Skip ignored directories  
        if should_ignore_directory "$source_file"; then
            continue
        fi
        
        # Calculate relative path and target file path
        local relative_path="${source_file#$source_dir/}"
        local target_file="$target_dir/$relative_path"
        
        handle_file "$source_file" "$target_file"
        
    done < <(find "$source_dir" -type f -print0)
}

# Function to validate directories
validate_directories() {
    # Check if we're in the right source directory
    if [[ ! -d ".claude" ]]; then
        print_color "$RED" "Error: .claude directory not found in current directory"
        print_color "$RED" "Please run this script from the root of your Claude Code project"
        exit 1
    fi
    
    # Check if target directory exists
    if [[ ! -d "$TARGET_DIR" ]]; then
        print_color "$RED" "Error: Target directory does not exist: $TARGET_DIR"
        exit 1
    fi
    
    # Check if target has .claude directory, create if not
    if [[ ! -d "$TARGET_DIR/.claude" ]]; then
        print_color "$YELLOW" "Creating .claude directory in target: $TARGET_DIR/.claude"
        if [[ "$DRY_RUN" == "false" ]]; then
            mkdir -p "$TARGET_DIR/.claude"
        fi
    fi
    
    # Check which source directories exist
    local missing_dirs=()
    for dir in "${SOURCE_DIRS[@]}"; do
        if [[ ! -d ".claude/$dir" ]]; then
            missing_dirs+=("$dir")
        fi
    done
    
    if [[ ${#missing_dirs[@]} -gt 0 ]]; then
        print_color "$YELLOW" "Warning: Some source directories don't exist: ${missing_dirs[*]}"
    fi
}

# Main function
main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -n|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -*)
                print_color "$RED" "Unknown option: $1"
                show_help
                exit 1
                ;;
            *)
                if [[ -z "$TARGET_DIR" ]]; then
                    TARGET_DIR="$1"
                else
                    print_color "$RED" "Error: Multiple target directories specified"
                    show_help
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # Check if target directory was provided
    if [[ -z "$TARGET_DIR" ]]; then
        print_color "$RED" "Error: Target directory not specified"
        show_help
        exit 1
    fi
    
    # Convert to absolute path
    TARGET_DIR="$(cd "$TARGET_DIR" && pwd)"
    
    # Validate directories
    validate_directories
    
    # Show summary
    echo
    print_color "$BLUE" "=== Claude Config Sync ==="
    echo "Source: $SCRIPT_DIR/.claude/"
    echo "Target: $TARGET_DIR/.claude/"
    if [[ "$DRY_RUN" == "true" ]]; then
        print_color "$YELLOW" "DRY RUN MODE - No changes will be made"
    fi
    echo
    
    # Create backup directory (only if not dry run and we might need it)
    create_backup_dir
    
    # Handle CLAUDE.md specially
    handle_claude_md
    
    # Sync each directory
    for dir in "${SOURCE_DIRS[@]}"; do
        local source_path=".claude/$dir"
        local target_path="$TARGET_DIR/.claude/$dir"
        
        if [[ -d "$source_path" ]]; then
            # Ensure target directory exists
            if [[ "$DRY_RUN" == "false" ]]; then
                mkdir -p "$target_path"
            fi
            sync_directory "$source_path" "$target_path"
        else
            print_color "$YELLOW" "Skipping missing source directory: $source_path"
        fi
    done
    
    echo
    if [[ "$DRY_RUN" == "true" ]]; then
        print_color "$BLUE" "=== Dry run completed ==="
        print_color "$BLUE" "Run without --dry-run to apply changes"
    else
        print_color "$GREEN" "=== Sync completed ==="
        if [[ -d "$BACKUP_DIR" ]] && [[ -n "$(ls -A "$BACKUP_DIR" 2>/dev/null)" ]]; then
            print_color "$GREEN" "Backups saved in: $BACKUP_DIR"
        fi
    fi
}

# Run main function with all arguments
main "$@"