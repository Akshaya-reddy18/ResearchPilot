import { LayoutDashboard, TrendingDown, AlertCircle, ShieldCheck, Brain, FileText, Settings } from 'lucide-react';

const menuItems = [
  { icon: LayoutDashboard, label: 'Dashboard', active: true },
  { icon: TrendingDown, label: 'Cost Analysis', active: false },
  { icon: AlertCircle, label: 'Anomaly Detection', active: false },
  { icon: ShieldCheck, label: 'Compliance Monitor', active: false },
  { icon: Brain, label: 'Risk Intelligence', active: false },
  { icon: FileText, label: 'Reports', active: false },
  { icon: Settings, label: 'Settings', active: false },
];

export function Sidebar() {
  return (
    <div className="w-64 h-screen bg-[#0B1220] border-r border-white/5 flex flex-col">
      {/* Logo */}
      <div className="p-6 border-b border-white/5">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center">
            <Brain className="w-6 h-6 text-white" />
          </div>
          <div>
            <div className="text-white font-semibold">CostOptimize</div>
            <div className="text-xs text-white/40">AI Platform</div>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4">
        {menuItems.map((item, index) => (
          <button
            key={index}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg mb-1 transition-all ${
              item.active
                ? 'bg-gradient-to-r from-blue-500/20 to-violet-600/20 text-blue-400 shadow-[0_0_20px_rgba(59,130,246,0.3)]'
                : 'text-white/60 hover:text-white/90 hover:bg-white/5'
            }`}
          >
            <item.icon className="w-5 h-5" />
            <span className="text-sm">{item.label}</span>
          </button>
        ))}
      </nav>
    </div>
  );
}
