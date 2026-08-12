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
      {/* TEMP: sidebar toggle disabled along with the sidebar itself, for
          local docker-compose smoke testing. Re-enable together with
          Sidebar in App.tsx. */}
      {false && (
        <button
          className="menu-btn"
          id="toggle-sidebar"
          aria-label="Toggle sidebar"
          onClick={onToggleSidebar}
        >
          ☰
        </button>
      )}
      <div className="chat-title">
        Causal Agent <span className="status-dot" aria-hidden="true" />
      </div>
      {/* TEMP: token-usage badge disabled for local docker-compose smoke
          testing. Re-enable before any real deploy. */}
      {false && (
        <span className="token-badge" id="token-tally-badge">
          {`${tokenTally.toLocaleString()} tokens used`}
        </span>
      )}

      {/* TEMP: login/profile-menu chrome disabled for local docker-compose
          smoke testing — this build has no auth flow. Re-enable before any
          real deploy. */}
      {false && (
        <div className="header-controls">
          {access.approved ? (
            <ProfileMenu access={access} onRequestTokens={onRequestTokens} />
          ) : (
            <button className="signin-btn" id="sign-in-btn" onClick={onLogin}>
              Login
            </button>
          )}
        </div>
      )}
    </header>
  );
}
