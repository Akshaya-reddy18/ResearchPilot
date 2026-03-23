import { ChevronDown, Download, Calendar } from 'lucide-react';

export function Header() {
  return (
    <div className="h-20 bg-[#0B1220]/80 backdrop-blur-xl border-b border-white/5 px-8 flex items-center justify-between">
      <div>
        <h1 className="text-2xl text-white">Enterprise Cost Intelligence</h1>
        <p className="text-sm text-white/50 mt-1">Real-time AI-powered insights & optimization</p>
      </div>

      <div className="flex items-center gap-4">
        {/* Company Selector */}
        <button className="flex items-center gap-2 px-4 py-2 bg-white/5 hover:bg-white/10 rounded-lg border border-white/10 transition-all">
          <span className="text-sm text-white">Acme Corp</span>
          <ChevronDown className="w-4 h-4 text-white/60" />
        </button>

        {/* Date Range */}
        <button className="flex items-center gap-2 px-4 py-2 bg-white/5 hover:bg-white/10 rounded-lg border border-white/10 transition-all">
          <Calendar className="w-4 h-4 text-white/60" />
          <span className="text-sm text-white">Last 30 Days</span>
          <ChevronDown className="w-4 h-4 text-white/60" />
        </button>

        {/* Export Button */}
        <button className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-blue-500 to-violet-600 hover:from-blue-600 hover:to-violet-700 rounded-lg transition-all shadow-lg shadow-blue-500/20">
          <Download className="w-4 h-4 text-white" />
          <span className="text-sm text-white">Export</span>
        </button>

        {/* User Avatar */}
        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center text-white cursor-pointer ring-2 ring-blue-500/20">
          <span className="text-sm">JD</span>
        </div>
      </div>
    </div>
  );
}
