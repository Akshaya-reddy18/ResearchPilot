import React from 'react'
import { LayoutDashboard, Layers, Cpu } from 'lucide-react'

export const Sidebar: React.FC = () => {
  return (
    <aside className="w-60 bg-white/60 dark:bg-slate-900/60 backdrop-blur rounded-lg p-4 shadow-sm hidden md:block">
      <div className="mb-6">
        <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-200">Navigation</h3>
        <nav className="mt-3 space-y-1">
          <a className="flex items-center gap-3 px-3 py-2 rounded hover:bg-slate-100 dark:hover:bg-slate-800 transition" href="#">
            <LayoutDashboard className="w-4 h-4 text-sky-600" />
            <span className="text-sm text-slate-700 dark:text-slate-200">Dashboard</span>
          </a>
          <a className="flex items-center gap-3 px-3 py-2 rounded hover:bg-slate-100 dark:hover:bg-slate-800 transition" href="#">
            <Layers className="w-4 h-4 text-sky-600" />
            <span className="text-sm text-slate-700 dark:text-slate-200">Workspaces</span>
          </a>
          <a className="flex items-center gap-3 px-3 py-2 rounded hover:bg-slate-100 dark:hover:bg-slate-800 transition" href="#">
            <Cpu className="w-4 h-4 text-sky-600" />
            <span className="text-sm text-slate-700 dark:text-slate-200">AI Tools</span>
          </a>
        </nav>
      </div>
    </aside>
  )
}

export default Sidebar
