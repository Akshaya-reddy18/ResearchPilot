import { FolderKanban, Plus, Calendar, FileText } from 'lucide-react';

const workspaces = [
  {
    id: 1,
    name: 'Machine Learning Research',
    papers: 12,
    lastModified: '2 hours ago',
    color: 'from-indigo-500 to-purple-600',
  },
  {
    id: 2,
    name: 'Natural Language Processing',
    papers: 8,
    lastModified: '1 day ago',
    color: 'from-blue-500 to-cyan-600',
  },
  {
    id: 3,
    name: 'Computer Vision',
    papers: 15,
    lastModified: '3 days ago',
    color: 'from-purple-500 to-pink-600',
  },
  {
    id: 4,
    name: 'Reinforcement Learning',
    papers: 6,
    lastModified: '5 days ago',
    color: 'from-orange-500 to-red-600',
  },
];

export function WorkspacesPage() {
  return (
    <div className="flex-1 overflow-y-auto bg-gradient-to-br from-[#0a0e27] to-[#1a1f4d]">
      <div className="p-8 max-w-7xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl text-white mb-2">Workspaces</h1>
            <p className="text-gray-400">Organize your research projects</p>
          </div>
          <button className="px-6 py-3 bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 text-white rounded-lg transition-all shadow-lg shadow-indigo-500/30 flex items-center gap-2">
            <Plus className="w-5 h-5" />
            New Workspace
          </button>
        </div>

        <div className="grid grid-cols-2 gap-6">
          {workspaces.map((workspace) => (
            <div
              key={workspace.id}
              className="bg-white/5 backdrop-blur-xl rounded-2xl p-6 border border-indigo-900/30 hover:bg-white/10 transition-all cursor-pointer group"
            >
              <div className="flex items-start justify-between mb-4">
                <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${workspace.color} flex items-center justify-center`}>
                  <FolderKanban className="w-6 h-6 text-white" />
                </div>
              </div>
              
              <h3 className="text-xl text-white mb-2 group-hover:text-indigo-300 transition-colors">
                {workspace.name}
              </h3>
              
              <div className="flex items-center gap-4 text-sm text-gray-400">
                <div className="flex items-center gap-1">
                  <FileText className="w-4 h-4" />
                  <span>{workspace.papers} papers</span>
                </div>
                <div className="flex items-center gap-1">
                  <Calendar className="w-4 h-4" />
                  <span>{workspace.lastModified}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
