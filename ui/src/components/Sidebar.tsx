import type { Conversation } from "../types";
import type { Theme } from "../lib/theme";

interface Props {
  collapsed: boolean;
  signedIn: boolean;
  conversations: Conversation[];
  hasMore: boolean;
  onSelect: (chatId: string) => void;
  onLoadMore: () => void;
  onNewChat: () => void;
  // Settings
  theme: Theme;
  onToggleTheme: () => void;
  model: string;
  onModelChange: (value: string) => void;
}

export function Sidebar({
  collapsed,
  signedIn,
  conversations,
  hasMore,
  onSelect,
  onLoadMore,
  onNewChat,
  theme,
  onToggleTheme,
  model,
  onModelChange,
}: Props) {
  return (
    <aside className={collapsed ? "sidebar collapsed" : "sidebar"} id="sidebar">
      <div className="brand">
        <span className="logo-mark" aria-hidden="true">
          <svg width="30" height="30" viewBox="0 0 30 30" fill="none">
            <defs>
              <linearGradient id="brand-grad" x1="0" y1="0" x2="30" y2="30">
                <stop offset="0" stopColor="#22d3ee" />
                <stop offset="1" stopColor="#8b5cf6" />
              </linearGradient>
            </defs>
            <circle cx="15" cy="15" r="11" stroke="url(#brand-grad)" strokeWidth="2.4" />
            <circle cx="15" cy="15" r="4" fill="url(#brand-grad)" />
            <path
              d="M15 1.5v6M15 22.5v6M1.5 15h6M22.5 15h6"
              stroke="url(#brand-grad)"
              strokeWidth="2"
              strokeLinecap="round"
            />
          </svg>
        </span>
        <span className="brand-name">
          CausalTrace<span className="brand-ai">AI</span>
        </span>
      </div>
      <div className="tracer-line" aria-hidden="true" />

      <button className="new-chat-btn" id="new-chat-btn" onClick={onNewChat}>
        <span className="plus">+</span> New chat
      </button>

      <nav className="history" aria-label="Recent workflows">
        <div className="history-title">Last 24 hours</div>
        <div id="history-items">
          {!signedIn ? (
            <div className="history-item hint" id="history-empty-hint">
              Log in to see your recent workflows
            </div>
          ) : conversations.length === 0 ? (
            <div className="history-item hint">No workflows in the last 24 hours</div>
          ) : (
            <>
              {conversations.map((conv) => (
                <div
                  className="history-item"
                  key={conv.chat_id}
                  title={conv.title}
                  onClick={() => onSelect(conv.chat_id)}
                >
                  {conv.title}
                </div>
              ))}
              {hasMore && (
                <div className="history-item hint history-load-more" onClick={onLoadMore}>
                  Load older
                </div>
              )}
            </>
          )}
        </div>
        {/* Stated where it's felt: conversations are swept after 24 hours, so
            this list is a rolling window rather than an archive. */}
        <div className="history-retention">Chats are deleted after 24 hours.</div>
      </nav>

      <div className="sidebar-settings">
        <div className="history-title">Settings</div>
        <div className="settings-panel">
          <label className="sidebar-setting">
            <span className="sw-label">Model</span>
            <select
              className="model-select sidebar-select"
              id="model-select"
              value={model}
              onChange={(e) => onModelChange(e.target.value)}
            >
              <option value="gemini-3.6-flash">gemini-3.6-flash</option>
              <option value="gemini-pro-latest">gemini-pro-latest</option>
            </select>
          </label>

          <label className="switch-group sidebar-switch" title="Toggle dark/light theme">
            <span className="sw-label">Theme</span>
            <button className="icon-btn theme-btn" id="theme-toggle" onClick={onToggleTheme}>
              {theme === "light" ? "☀" : "☾"}
            </button>
          </label>
        </div>
      </div>
    </aside>
  );
}
