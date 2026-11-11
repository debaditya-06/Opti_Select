import { useState } from 'react';
import { Link, NavLink } from 'react-router-dom';
import type { ReactNode } from 'react';
import type { BreadcrumbItem } from '../types';
import { Breadcrumbs } from './Breadcrumbs';

interface LayoutProps {
  children: ReactNode;
  breadcrumbs?: BreadcrumbItem[];
}

const NAV_LINKS = [
  { to: '/', label: 'Home', end: true },
  { to: '/experiment', label: 'Run Experiment', end: false },
  { to: '/datasets', label: 'Datasets', end: false },
  { to: '/methodology', label: 'Methodology', end: false },
  { to: '/results', label: 'Results', end: false },
];

export function Layout({ children, breadcrumbs }: LayoutProps) {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">
        Skip to main content
      </a>

      <header className="site-header">
        <div className="container header-inner">
          <Link to="/" className="brand-link" onClick={() => setMenuOpen(false)}>
            <svg
              width="28"
              height="28"
              viewBox="0 0 32 32"
              aria-hidden="true"
            >
              <rect width="32" height="32" rx="6" fill="#0E9F8E" />
              <circle cx="8" cy="10" r="3" fill="#EAF4F4" opacity="0.5" />
              <circle cx="16" cy="8" r="3" fill="#FFFFFF" stroke="#12232E" strokeWidth="1" />
              <circle cx="24" cy="12" r="3" fill="#FFFFFF" stroke="#12232E" strokeWidth="1" />
              <circle cx="10" cy="20" r="3" fill="#EAF4F4" opacity="0.5" />
              <circle cx="20" cy="22" r="3" fill="#FFFFFF" stroke="#12232E" strokeWidth="1" />
            </svg>
            <span>OptiSelect</span>
            <span className="brand-subtitle">| Feature Selection</span>
          </Link>

          <button
            type="button"
            className="menu-toggle"
            aria-expanded={menuOpen}
            aria-controls="primary-navigation"
            onClick={() => setMenuOpen((prev) => !prev)}
          >
            {menuOpen ? 'Close menu' : 'Menu'}
          </button>

          <nav
            id="primary-navigation"
            className={`primary-nav${menuOpen ? ' is-open' : ''}`}
            aria-label="Primary navigation"
          >
            <ul className="nav-list">
              {NAV_LINKS.map((item) => (
                <li key={item.to}>
                  <NavLink
                    to={item.to}
                    end={item.end}
                    className="nav-link"
                    onClick={() => setMenuOpen(false)}
                  >
                    {item.label}
                  </NavLink>
                </li>
              ))}
            </ul>
          </nav>
        </div>
      </header>

      <main id="main-content" className="site-main" tabIndex={-1}>
        <div className="container">
          {breadcrumbs && breadcrumbs.length > 0 && <Breadcrumbs items={breadcrumbs} />}
          {children}
        </div>
      </main>

      <footer className="site-footer">
        <div className="container footer-inner">
          <div>
            <strong>OptiSelect</strong> — Group 1 Optimization Project. Leakage-free feature selection evaluation.
          </div>
          <ul className="footer-links">
            <li><Link to="/experiment">Experiment Runner</Link></li>
            <li><Link to="/datasets">Datasets</Link></li>
            <li><Link to="/methodology">Methodology</Link></li>
            <li><Link to="/results">Final Results</Link></li>
          </ul>
        </div>
      </footer>
    </div>
  );
}
