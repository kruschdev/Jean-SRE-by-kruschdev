import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import path from "path";
import os from "os";

class McpClientManager {
  constructor() {
    this.clients = {}; // Map of name -> { client, tools }
  }

  async connect(name, command, args, env = {}, cwd = undefined, timeoutMs = 6000) {
    if (this.clients[name]) return; // Already connected

    let transport = null;
    let timer = null;
    let timedOut = false;

    const connectPromise = (async () => {
      transport = new StdioClientTransport({
        command,
        args,
        env: { ...process.env, ...env },
        ...(cwd ? { cwd } : {})
      });

      const client = new Client(
        { name: `hermes-harness-${name}`, version: "1.0.0" },
        { capabilities: {} }
      );

      await client.connect(transport);
      if (timedOut) {
        try { await client.close(); } catch {}
        return;
      }

      // Fetch tools
      const toolsResponse = await client.listTools();
      const tools = toolsResponse.tools;

      this.clients[name] = { client, tools };
      console.log(`[MCP] Connected to ${name} (${tools.length} tools available)`);
    })();

    const timeoutPromise = new Promise((_, reject) => {
      timer = setTimeout(() => {
        timedOut = true;
        if (transport && typeof transport.close === 'function') {
          try { transport.close(); } catch {}
        }
        reject(new Error(`Connection to ${name} timed out after ${timeoutMs}ms`));
      }, timeoutMs);
    });

    try {
      await Promise.race([connectPromise, timeoutPromise]);
    } finally {
      if (timer) clearTimeout(timer);
    }
  }

  getAllTools() {
    const allTools = [];
    for (const [serverName, serverData] of Object.entries(this.clients)) {
      for (const t of serverData.tools) {
        const dashedName = t.name.replace(/_/g, '-');
        const exposedName = dashedName.startsWith(`${serverName}-`)
          ? dashedName
          : `${serverName}-${dashedName}`;
        allTools.push({
          type: 'function',
          function: {
            name: exposedName,
            description: t.description || `Tool ${t.name} from ${serverName}`,
            parameters: t.inputSchema || { type: 'object', properties: {} }
          }
        });
      }
    }
    return allTools;
  }

  async callTool(fullToolName, args) {
    // Intercept homelab-memory or krusch-memory search/add calls and map them to krusch-context-mcp
    const searchAliases = [
      'mcp_homelab-memory_search',
      'mcp-homelab-memory-search',
      'homelab-memory-search',
      'krusch-memory-search',
      'krusch-memory-search-memory',
      'krusch-context-search-memory'
    ];
    const addAliases = [
      'mcp_homelab-memory_add',
      'mcp-homelab-memory-add',
      'homelab-memory-add',
      'krusch-memory-add',
      'krusch-memory-add-memory',
      'krusch-context-add-memory'
    ];
    const nuggetsNudgesAliases = [
      'nuggets-nudges',
      'mcp_nuggets-memory_nudges',
      'nuggets-memory-nudges',
      'krusch-context-nuggets-nudges'
    ];
    const nuggetsRememberAliases = [
      'nuggets-remember',
      'mcp_nuggets-memory_remember',
      'nuggets-memory-remember',
      'krusch-context-nuggets-remember'
    ];

    if (searchAliases.includes(fullToolName)) {
      console.warn(`[MCP Alias] Intercepted legacy search tool call '${fullToolName}'. Mapping to 'krusch-context-krusch-context-search-memory'.`);
      fullToolName = 'krusch-context-krusch-context-search-memory';
    } else if (addAliases.includes(fullToolName)) {
      console.warn(`[MCP Alias] Intercepted legacy add tool call '${fullToolName}'. Mapping to 'krusch-context-krusch-context-add-memory'.`);
      fullToolName = 'krusch-context-krusch-context-add-memory';
    } else if (nuggetsNudgesAliases.includes(fullToolName)) {
      console.warn(`[MCP Alias] Intercepted legacy nuggets nudges tool call '${fullToolName}'. Mapping to 'krusch-context-krusch-context-nugget-nudges'.`);
      fullToolName = 'krusch-context-krusch-context-nugget-nudges';
      if (!args.query) {
        args.query = 'jean-sre';
      }
      if (!args.active_project) {
        args.active_project = args.project || 'jean-sre';
      }
    } else if (nuggetsRememberAliases.includes(fullToolName)) {
      console.warn(`[MCP Alias] Intercepted legacy nuggets remember tool call '${fullToolName}'. Mapping to 'krusch-context-krusch-context-nugget-remember'.`);
      fullToolName = 'krusch-context-krusch-context-nugget-remember';
      if (!args.active_project) {
        args.active_project = args.project || 'jean-sre';
      }
    } else if (['krusch_fs_search', 'krusch_fs_locate', 'krusch_fs_page_tree', 'krusch_fs_read_page', 'krusch_fs_remember_on_page', 'krusch_fs_sync'].includes(fullToolName)) {
      fullToolName = `krusch-fs-${fullToolName.replace(/_/g, '-')}`;
    } else if (['krusch_git_list_repos', 'krusch_git_read_tree', 'krusch_git_read_blob', 'krusch_git_semantic_search', 'krusch_git_search_symbols', 'krusch_git_file_symbols', 'krusch_git_dependency_graph'].includes(fullToolName)) {
      fullToolName = `krusch-git-${fullToolName.replace(/_/g, '-')}`;
    } else if (['query_design_rules', 'analyze_layout_structure', 'css_to_tailwind', 'generate_color_palette', 'get_canonical_template', 'list_canonical_templates'].includes(fullToolName)) {
      fullToolName = `krusch-frontend-${fullToolName.replace(/_/g, '-')}`;
    } else if (['krusch_context_retrieve', 'krusch_context_remember', 'krusch_context_revise', 'krusch_context_nudge', 'krusch_context_health'].includes(fullToolName)) {
      fullToolName = `krusch-context-${fullToolName.replace(/_/g, '-')}`;
    }

    // Intercept legacy get-memory-priorities-index calls and route them to krusch_context_list_memories
    const legacyNames = [
      'get-memory-priorities-index',
      'get_memory_priorities_index',
      'krusch-context-get-memory-priorities-index',
      'krusch-context-get_memory_priorities_index',
      'krusch-memory-get-memory-priorities-index',
      'krusch-memory-get_memory_priorities_index'
    ];
    
    if (legacyNames.includes(fullToolName)) {
      console.warn(`[MCP Alias] Intercepted legacy tool call '${fullToolName}'. Mapping to 'krusch-context-krusch-context-list-memories'.`);
      fullToolName = 'krusch-context-krusch-context-list-memories';
      args = {
        category: 'priorities',
        project: args.project || 'jean-sre',
        limit: args.limit || 5
      };
    }

    // Match against registered server names (longest prefix wins)
    const normalizedInput = fullToolName.replace(/_/g, '-');
    let serverName = null;
    let convertedToolName = null;
    
    for (const name of Object.keys(this.clients)) {
      if (normalizedInput.startsWith(name + '-')) {
        if (!serverName || name.length > serverName.length) {
          serverName = name;
          convertedToolName = normalizedInput.substring(name.length + 1);
        }
      }
    }
    
    if (!serverName) {
      throw new Error(`No registered MCP server matches tool: ${fullToolName}`);
    }

    const { client, tools } = this.clients[serverName];
    const serverUnderscore = serverName.replace(/-/g, '_');
    // Match whether caller used single prefix (krusch-git-search-symbols) or double prefix (krusch-git-krusch-git-search-symbols)
    const originalTool = tools.find(t => {
      const tDashed = t.name.replace(/_/g, '-');
      return (
        tDashed === convertedToolName ||
        tDashed === `${serverName}-${convertedToolName}` ||
        t.name === `${serverUnderscore}_${convertedToolName.replace(/-/g, '_')}`
      );
    });
    const toolName = originalTool ? originalTool.name : convertedToolName.replace(/-/g, '_');

    // Normalize content parameter for krusch-context memory write tools
    if (serverName === 'krusch-context' && ['krusch_context_remember', 'krusch_context_add_memory', 'krusch_context_nugget_remember'].includes(toolName)) {
      if (!args.content || String(args.content).trim() === '') {
        args.content = args.value || args.text || args.note || args.summary || args.description || (args.key ? `Updated key ${args.key}` : 'Recorded SRE observation.');
      }
      if (toolName === 'krusch_context_nugget_remember' && !args.value) {
        args.value = args.content;
      }
    }

    const result = await client.callTool({
      name: toolName,
      arguments: args
    });

    if (result.isError) {
        return `Error: ${result.content[0].text}`;
    }
    return result.content[0].text;
  }

  async disconnectAll() {
    for (const data of Object.values(this.clients)) {
      await data.client.close();
    }
  }
}

// Singleton instance
const mcpManager = new McpClientManager();

import { fileURLToPath } from 'url';
import fs from 'fs';

let _initPromise = null;

export async function initMcpClients() {
    if (_initPromise) return _initPromise;
    _initPromise = _doInitMcpClients();
    return _initPromise;
}

async function _doInitMcpClients() {
    const __dirname = path.dirname(fileURLToPath(import.meta.url));
    const appRoot = process.env.APP_ROOT || path.resolve(__dirname, '..');
    const homelabPath = process.env.HOMELAB_PATH || path.join(os.homedir(), 'homelab');
    
    // 1. Context / Working Memory Server (Tier 1)
    const contextPath = path.resolve(appRoot, '../krusch-context-mcp/src/index.js');
    const defaultDbUrl = process.env.DATABASE_URL || process.env.DBOS_DATABASE_URL || '';
    try {
      await mcpManager.connect(
        'krusch-context', 
        'node', 
        [contextPath],
        { 
          ...(defaultDbUrl ? { DBOS_DATABASE_URL: defaultDbUrl, DATABASE_URL: defaultDbUrl } : {}),
          OLLAMA_URL: process.env.OLLAMA_EMBED_HOST || 'http://127.0.0.1:11434',
          OLLAMA_NODES: process.env.OLLAMA_EMBED_HOST || 'http://127.0.0.1:11434',
          TAG_MODEL: 'qwen2.5-coder:7b'
        },
        appRoot
      );
    } catch (e) {
      console.error(`[MCP] Krusch Context failed to connect: ${e.message}`);
    }

    // 2. KruschGit (Tier 2A Codebase DAG, AST & Hybrid RRF Semantic Search)
    const gitCandidates = [
      path.resolve(appRoot, '../krusch-git/server/mcp.js'),
      path.resolve(appRoot, '../pg-git/server/mcp.js')
    ];
    const gitPath = gitCandidates.find(p => fs.existsSync(p)) || gitCandidates[0];
    try {
      await mcpManager.connect(
        'krusch-git',
        'node',
        [gitPath],
        {
          ...(defaultDbUrl ? { DATABASE_URL: defaultDbUrl } : {}),
          OLLAMA_URL: process.env.OLLAMA_EMBED_HOST || 'http://127.0.0.1:11434',
          OLLAMA_NODES: process.env.OLLAMA_EMBED_HOST || 'http://127.0.0.1:11434',
          PORT: ''
        },
        path.dirname(path.dirname(gitPath))
      );
    } catch (e) {
      console.error(`[MCP] KruschGit failed to connect: ${e.message}`);
    }

    // 3. Infrastructure Compose & Diagnostics Server
    const infraPath = path.resolve(appRoot, '../krusch-infra-mcp/server.js');
    const defaultAllowedPaths = [
      `${homelabPath}/projects`,
      '/app/projects',
      '/home/agent/homelab/projects',
      '/home/kruschdev/homelab/projects',
      '/home/krusch/homelab/projects'
    ].join(':');
    try {
      await mcpManager.connect(
        'krusch-infra',
        'node',
        [infraPath],
        { 
            ALLOWED_PROJECT_PATHS: process.env.ALLOWED_PROJECT_PATHS || defaultAllowedPaths,
            PORT: '' // Prevent infra server from attempting SSE mode
        },
        path.dirname(infraPath)
      );
    } catch (e) {
      console.error(`[MCP] Krusch Infra failed to connect: ${e.message}`);
    }

    // 4. Sentinel Multi-Node Telemetry & FRAXIS Self-State Security Server
    const sentinelPath = path.resolve(appRoot, '../krusch-sentinel-mcp/dist/index.js');
    try {
      await mcpManager.connect(
        'sentinel',
        'node',
        [sentinelPath],
        { PORT: '' },
        path.dirname(sentinelPath)
      );
    } catch (e) {
      console.error(`[MCP] Sentinel telemetry failed to connect: ${e.message}`);
    }

    // 5. KruschFS (Tier 2B OS Filesystem & Multi-Node Fleet Page Memory)
    const libPyPath = path.resolve(appRoot, '../../lib-py');
    const fsSrcPath = path.resolve(appRoot, '../krusch-fs/src');
    const nexusSrcPath = path.resolve(appRoot, '../krusch-nexus/src');
    try {
      const pythonBin = process.env.PYTHON_BIN || 'python3';
      await mcpManager.connect(
        'krusch-fs',
        pythonBin,
        ['-m', 'krusch_fs.mcp'],
        {
          PYTHONPATH: [fsSrcPath, nexusSrcPath, libPyPath].join(':'),
          OLLAMA_URL: process.env.OLLAMA_EMBED_HOST || 'http://127.0.0.1:11434',
          PORT: ''
        },
        appRoot
      );
    } catch (e) {
      console.error(`[MCP] KruschFS failed to connect: ${e.message}`);
    }

    // 6. Krusch Frontend Canonical Design Server
    const frontendPath = path.resolve(appRoot, '../krusch-frontend-mcp/src/index.js');
    try {
      await mcpManager.connect(
        'krusch-frontend',
        'node',
        [frontendPath],
        { PORT: '' },
        path.dirname(frontendPath)
      );
    } catch (e) {
      console.error(`[MCP] Krusch Frontend failed to connect: ${e.message}`);
    }

    return mcpManager;
}

export async function getMcpTools() {
    await initMcpClients();
    return mcpManager.getAllTools();
}

export async function callMcpTool(name, args) {
    await initMcpClients();
    return await mcpManager.callTool(name, args);
}

export function getConnectedServers() {
    return Object.keys(mcpManager.clients);
}
