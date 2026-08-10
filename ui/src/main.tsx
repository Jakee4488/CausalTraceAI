import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "highlight.js/styles/atom-one-dark.css";
import "./styles.css";

const root = document.getElementById("root");
if (!root) throw new Error("#root is missing from index.html");

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
