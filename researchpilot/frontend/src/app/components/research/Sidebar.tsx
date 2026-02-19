import { Home, Upload, FolderKanban, Sparkles } from 'lucide-react';
import { Link, useLocation } from 'react-router-dom';

const menuItems = [
  { icon: Home, label: 'Dashboard', path: '/dashboard' },
  { icon: Upload, label: 'Upload PDF', path: '/upload' },
  { icon: FolderKanban, label: 'Workspaces', path: '/workspaces' },
  { icon: Sparkles, label: 'AI Tools', path: '/chat' },
];

export function Sidebar() {
  const location = useLocation();

  return (
    <div className="w-64 h-screen bg-[#0a0e27] border-r border-indigo-900/30 flex flex-col">
      {/* Logo */}
      <div className="p-6 border-b border-indigo-900/30">
        <Link to="/" className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
            <Sparkles className="w-6 h-6 text-white" />
          </div>
          <div>
            <div className="text-white font-semibold text-lg">ResearchPilot</div>
            <div className="text-xs text-indigo-400">AI Research Hub</div>
          </div>
        </Link>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4">
        {menuItems.map((item, index) => {
          const isActive = location.pathname === item.path;
          return (
            <Link
              key={index}
              to={item.path}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg mb-1 transition-all ${
                isActive
                  ? 'bg-gradient-to-r from-indigo-500/20 to-purple-600/20 text-indigo-300'
                  : 'text-gray-400 hover:text-white hover:bg-white/5'
              }`}
            >
              <item.icon className="w-5 h-5" />
              <span className="text-sm">{item.label}</span>
            </Link>
          );
        })}
      </nav>
    </div>
  );
}
