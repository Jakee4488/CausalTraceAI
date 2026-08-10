import type { Access } from "../hooks/useAccess";
import { ProfileMenu } from "./ProfileMenu";

interface Props {
  onToggleSidebar: () => void;
  tokenTally: number;
  access: Access;
  onLogin: () => void;
  onRequestTokens: () => void;
}

export function ChatHeader({
  onToggleSidebar,
  tokenTally,
  access,
  onLogin,
  onRequestTokens,
}: Props) {
  return (
    <header className="chat-header">
      <button
        className="menu-btn"
        id="toggle-sidebar"
        aria-label="Toggle sidebar"
        onClick={onToggleSidebar}
      >
        ☰
      </button>
      <div className="chat-title">
        Causal Agent <span className="status-dot" aria-hidden="true" />
      </div>
      <span className="token-badge" id="token-tally-badge">
        {`${tokenTally.toLocaleString()} tokens used`}
      </span>

      <div className="header-controls">
        {access.approved ? (
          <ProfileMenu access={access} onRequestTokens={onRequestTokens} />
        ) : (
          <button className="signin-btn" id="sign-in-btn" onClick={onLogin}>
            Login
          </button>
        )}
      </div>
    </header>
  );
}
