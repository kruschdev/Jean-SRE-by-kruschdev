export function parseThinkerPlan(thinkerOutput) {
    const intents = [];

    // Strategy 1: Extract from ### INVESTIGATION PLAN section
    const planMatch = thinkerOutput.match(/### INVESTIGATION PLAN[\s\S]*?(?=###|$)/i);
    const searchText = planMatch ? planMatch[0] : thinkerOutput;

    // Match numbered items flexibly: "1. anything..." until the next number
    const numberedItems = searchText.match(/(?:\n|^)\d+\.\s+[\s\S]*?(?=(?:\n\d+\.\s+)|$)/g) || [];
    for (const item of numberedItems) {
        const cleanItem = item.trim().replace(/\n/g, ' ');
        if (cleanItem.length > 5) {
            intents.push({ task: cleanItem.slice(0, 500), raw: cleanItem });
        }
    }

    // Strategy 2: Fallback — extract backtick-wrapped shell commands, AST queries, MCP tools, or file paths
    if (intents.length === 0) {
        const cmdMatches = thinkerOutput.matchAll(/`((?:read_file|search_symbols|dependency_graph|krusch_git_search_symbols|krusch_git_dependency_graph|krusch_git_semantic_search|krusch_git_file_symbols|krusch_fs_search|krusch_fs_locate|krusch_fs_page_tree|krusch_fs_read_page|query_design_rules|analyze_layout_structure|audit_agent_self_state|verify_state_integrity|detect_memory_anomaly|grep|find|cat|head|wc|ls|run_shell_command)(?:\s+|\()[^`]+)`/g);
        for (const m of cmdMatches) {
            const cmd = m[1].trim();
            if (cmd.length > 6 && intents.length < 4) {
                const isDirectTool = /^(?:search_symbols|dependency_graph|krusch_git_|krusch_fs_|query_design_rules|analyze_layout_structure|audit_agent_self_state|verify_state_integrity|detect_memory_anomaly)\b/i.test(cmd);
                intents.push({ task: isDirectTool ? cmd : `run_shell_command \`${cmd}\``, raw: cmd });
            }
        }
    }

    // Strategy 3: Extract file paths that look like audit targets
    if (intents.length === 0) {
        const pathMatches = thinkerOutput.matchAll(/`(\/home\/(?:krusch|kruschdev|agent)\/homelab\/projects\/[^`]+)`/g);
        const seen = new Set();
        for (const m of pathMatches) {
            const p = m[1].trim();
            if (!seen.has(p) && intents.length < 3) {
                seen.add(p);
                intents.push({ task: `read_file \`${p}\``, raw: p });
            }
        }
    }

    // Strategy 4: Handle split-backtick format — `action` `command` as separate groups
    if (intents.length === 0) {
        const splitMatches = thinkerOutput.matchAll(/`(read_file|run_shell_command|search_symbols|dependency_graph|krusch_git_search_symbols|krusch_git_dependency_graph|krusch_git_semantic_search|krusch_git_file_symbols|krusch_fs_search|krusch_fs_locate|krusch_fs_page_tree|krusch_fs_read_page|query_design_rules|analyze_layout_structure|audit_agent_self_state|verify_state_integrity|detect_memory_anomaly)`\s*`([^`]+)`/g);
        for (const m of splitMatches) {
            const action = m[1].trim();
            const cmd = m[2].trim();
            if (cmd.length > 2 && intents.length < 4) {
                intents.push({ task: `${action} \`${cmd}\``, raw: cmd });
            }
        }
    }

    return intents;
}
