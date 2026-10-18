import { Activity, AlertTriangle, ClipboardList, LayoutDashboard, Menu, Plus, ShieldCheck, Users, X } from "lucide-react";
import { useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { useHealth } from "../hooks/useApi";

const links = [
  { to: "/", label: "Visão geral", icon: LayoutDashboard, end: true },
  { to: "/patients", label: "Pacientes", icon: Users, end: false },
];

export function AppShell() {
  const health = useHealth();
  const [menuOpen, setMenuOpen] = useState(false);
  const location = useLocation();
  const pageTitle = location.pathname.startsWith("/patients/new")
    ? "Novo paciente"
    : location.pathname.startsWith("/patients/")
      ? "Acompanhamento do paciente"
      : location.pathname.startsWith("/patients")
        ? "Pacientes"
        : "Visão geral";

  const statusLabel = health.isPending
    ? "Verificando API"
    : health.isError
      ? "API indisponível"
      : health.data?.model_ready
        ? "API e modelo prontos"
        : "API disponível · modelo indisponível";

  return (
    <div className="app-frame">
      <button className="mobile-menu-trigger" onClick={() => setMenuOpen(true)} aria-label="Abrir navegação">
        <Menu size={20} />
      </button>
      {menuOpen && <button className="mobile-scrim" aria-label="Fechar navegação" onClick={() => setMenuOpen(false)} />}
      <aside className={`sidebar ${menuOpen ? "sidebar-open" : ""}`}>
        <div className="brand">
          <span className="brand-mark"><Activity size={18} strokeWidth={2.5} /></span>
          <span><strong>Vigília</strong><small>Risco experimental</small></span>
          <button className="sidebar-close" onClick={() => setMenuOpen(false)} aria-label="Fechar menu"><X size={18} /></button>
        </div>
        <div className="sidebar-label">ACOMPANHAMENTO</div>
        <nav aria-label="Navegação principal">
          {links.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              onClick={() => setMenuOpen(false)}
              className={({ isActive }) => `nav-link ${isActive ? "nav-link-active" : ""}`}
            >
              <Icon size={17} />
              <span>{label}</span>
            </NavLink>
          ))}
          <NavLink
            to="/patients/new"
            onClick={() => setMenuOpen(false)}
            className={({ isActive }) => `nav-link ${isActive ? "nav-link-active" : ""}`}
          >
            <Plus size={17} />
            <span>Cadastrar paciente</span>
          </NavLink>
        </nav>
        <div className="sidebar-bottom">
          <div className="sidebar-note-icon"><ShieldCheck size={16} /></div>
          <p>Ambiente experimental local. Não utilizar para decisões clínicas.</p>
          <div className="sidebar-version"><ClipboardList size={14} /> API longitudinal</div>
        </div>
      </aside>

      <div className="main-column">
        <header className="topbar">
          <div className="topbar-title">
            <span className="eyebrow">SISTEMA EXPERIMENTAL DE APOIO À DECISÃO</span>
            <h1>{pageTitle}</h1>
          </div>
          <div className={`connection-pill ${health.isError ? "connection-offline" : health.data?.model_ready ? "connection-ready" : ""}`}>
            {health.isError ? <AlertTriangle size={15} /> : <span className="connection-dot" />}
            <span>{statusLabel}</span>
          </div>
        </header>
        <main className="page-content"><Outlet /></main>
        <footer className="app-footer">
          <span>Interface de demonstração · Dados inseridos manualmente</span>
          <span>Não validado para decisões clínicas</span>
        </footer>
      </div>
    </div>
  );
}
