"""Coverage for the declarative heuristic intent rule table.

Every IntentRule in core.dual_process.tool_loop.INTENT_RULES must have a
canonical example utterance here that reaches it — matches one of its
patterns, survives argument extraction, and is not captured by an earlier
rule — with every rule's tool available. A rule that cannot be reached is
dead (shadowed or unmatchable) and fails these tests; a new rule without
an example fails them too.
"""

import pytest

from core.dual_process.tool_loop import (
    INTENT_RULES,
    detect_tool_intent,
    match_intent_rule,
)

ALL_RULE_TOOLS = {rule.tool for rule in INTENT_RULES}

# One canonical utterance per rule, by rule name. Derived from the original
# regexes and the existing phase2/phase6 detection tests.
RULE_EXAMPLES: dict[str, str] = {
    # Skills
    "skills_list": "list the imported skills",
    "skills_enable": "enable the skill weather-fetcher",
    "skills_disable": "disable the skill weather-fetcher",
    "skills_delete": "remove the skill weather-fetcher",
    "skills_audit": "audit the skill registry",
    "skills_audit_problem": "what's the problem with the skill import?",
    "skills_create": "create a skill that summarizes reports",
    "skills_make_permanent": "make this permanent",
    # Filesystem
    "fs_read_file": "read file /tmp/test.txt",
    "fs_show_file": "show the contents of config.yaml",
    "fs_list_dir_path": "list files in /home/user",
    "fs_ls": "ls /var/log",
    "fs_whats_in_path": "what's in /home/projects",
    "fs_list_files_default": "list the system files",
    "fs_show_files_default": "show me the files",
    "fs_what_files_here": "what files are available here",
    "fs_search_quoted_in_path": "search for 'TODO' in /src",
    "fs_grep": "grep main /src/app.py",
    "fs_glob_in_path": "find files matching '*.py' in /src",
    "fs_write_quoted_to_path": "write 'hello world' to file /tmp/out.txt",
    "fs_create_file_with_content": "create a file /tmp/out.txt with 'hello'",
    "fs_delete_path": "delete the file /tmp/junk.txt",
    "fs_rm": "rm /tmp/junk.txt",
    # System inspection
    "system_memory_usage_query": "what's the current memory usage",
    "system_memory_how_much": "how much memory is left",
    "system_free_command": "free -m",
    "system_memory_status": "show memory info",
    # Shell
    "shell_hostname_question": "what's the hostname of this machine",
    "shell_machine_name_running": "the name of the machine this is running on",
    "shell_hostname_bare": "hostname",
    "shell_run_known_command": "run the command uname -a",
    "shell_uname_bare": "uname -a",
    "shell_cat_etc_hostname": "cat /etc/hostname",
    "shell_disk_space_question": "how much disk space is left",
    "shell_df_bare": "df -h /",
    "shell_run_quoted": "run the command 'echo hello'",
    "shell_execute_quoted": "execute 'echo hi'",
    "shell_run_explicit": "run apt update",
    "shell_do_explicit": "do arbitrary-tool --flag",
    # Web fetch/extract
    "web_fetch_url": "fetch https://example.com",
    "web_extract_text_url": "extract the text from https://example.com/article",
    "web_get_text_url": "get the readable text of https://example.com",
    # Web search
    "web_search_the_web": "search the web for 'python tutorial'",
    "web_search_online_suffix": "search for python decorators online",
    "web_look_up": "look up the eiffel tower",
    "web_google": "google rust borrow checker",
    "web_browse_for": "browse the web for cat pictures",
    "web_find_about_online": "find info about quantum computing online",
    "web_released_since": "list games released after 2024",
    "web_top_best_released": "top laptops released in 2025",
    "web_latest_news": "latest news about the election",
    "web_news_about": "news about the local elections",
    "web_what_happened_recently": "what happened with the launch today",
    "web_current_facts": "current price of bitcoin",
    "web_who_is_current": "who is the current president of france",
    "web_what_is_latest": "what is the current version of python",
    "web_check_latest": "check the latest exchange rates",
    # Browser
    "browser_open_url": "open https://example.com in the browser",
    "browser_browse_to_url": "browse to https://example.com",
    "browser_get_page_text": "get the page text from https://example.com",
    "browser_screenshot_url": "screenshot of https://example.com",
    "browser_take_screenshot": "take a screenshot",
    # Calendar
    "calendar_list_events": "show my events for 2026-04-02",
    "calendar_check_calendar": "check my calendar for 2026-04-03",
    "calendar_create_event": "create event 'Team standup' on 2026-04-02T09:00",
}


class TestIntentRuleTable:
    def test_rule_names_are_unique(self):
        names = [rule.name for rule in INTENT_RULES]
        assert len(names) == len(set(names))

    def test_every_rule_has_at_least_one_pattern(self):
        for rule in INTENT_RULES:
            assert rule.patterns, f"rule {rule.name} has no patterns"

    def test_examples_cover_exactly_the_rule_table(self):
        assert set(RULE_EXAMPLES) == {rule.name for rule in INTENT_RULES}


class TestIntentRuleCoverage:
    @pytest.mark.parametrize("rule", INTENT_RULES, ids=lambda r: r.name)
    def test_example_reaches_its_rule(self, rule):
        example = RULE_EXAMPLES[rule.name]
        matched = match_intent_rule(example, ALL_RULE_TOOLS)
        assert matched is not None, (
            f"rule {rule.name}: example {example!r} matched no rule"
        )
        hit, args = matched
        assert hit.name == rule.name, (
            f"rule {rule.name}: example {example!r} was captured by "
            f"earlier rule {hit.name}"
        )
        intent = detect_tool_intent(example, ALL_RULE_TOOLS)
        assert intent is not None
        assert intent.tool_name == rule.tool
        assert intent.arguments == args
