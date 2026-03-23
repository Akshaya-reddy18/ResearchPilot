// import { FolderKanban, Plus, Calendar, FileText } from 'lucide-react';
// import Editor from './Editor';

// const workspaces = [
//   {
//     id: 1,
//     name: 'Machine Learning Research',
//     papers: 12,
//     lastModified: '2 hours ago',
//     color: 'from-indigo-500 to-purple-600',
//   },
//   {
//     id: 2,
//     name: 'Natural Language Processing',
//     papers: 8,
//     lastModified: '1 day ago',
//     color: 'from-blue-500 to-cyan-600',
//   },
//   {
//     id: 3,
//     name: 'Computer Vision',
//     papers: 15,
//     lastModified: '3 days ago',
//     color: 'from-purple-500 to-pink-600',
//   },
//   {
//     id: 4,
//     name: 'Reinforcement Learning',
//     papers: 6,
//     lastModified: '5 days ago',
//     color: 'from-orange-500 to-red-600',
//   },
// ];

// export function WorkspacesPage() {
//   return (
//     <div className="flex-1 overflow-y-auto bg-gradient-to-br from-[#0a0e27] to-[#1a1f4d]">
//       <div className="p-8 max-w-7xl mx-auto">
//         <div className="flex items-center justify-between mb-8">
//           <div>
//             <h1 className="text-3xl text-white mb-2">Workspaces</h1>
//             <p className="text-gray-400">Organize your research projects</p>
//           </div>
//           <button className="px-6 py-3 bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 text-white rounded-lg transition-all shadow-lg shadow-indigo-500/30 flex items-center gap-2">
//             <Plus className="w-5 h-5" />
//             New Workspace
//           </button>
//         </div>

//         <div className="grid grid-cols-2 gap-6">
//           {workspaces.map((workspace) => (
//             <div
//               key={workspace.id}
//               className="bg-white/5 backdrop-blur-xl rounded-2xl p-6 border border-indigo-900/30 hover:bg-white/10 transition-all cursor-pointer group"
//             >
//               <div className="flex items-start justify-between mb-4">
//                 <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${workspace.color} flex items-center justify-center`}>
//                   <FolderKanban className="w-6 h-6 text-white" />
//                 </div>
//               </div>
              
//               <h3 className="text-xl text-white mb-2 group-hover:text-indigo-300 transition-colors">
//                 {workspace.name}
//               </h3>
              
//               <div className="flex items-center gap-4 text-sm text-gray-400">
//                 <div className="flex items-center gap-1">
//                   <FileText className="w-4 h-4" />
//                   <span>{workspace.papers} papers</span>
//                 </div>
//                 <div className="flex items-center gap-1">
//                   <Calendar className="w-4 h-4" />
//                   <span>{workspace.lastModified}</span>
//                 </div>
//               </div>
//             </div>
//           ))}
//         </div>
        
//         {/* Example editor embedded for quick testing */}
//         <div className="mt-8">
//           <div className="bg-white/5 backdrop-blur-xl rounded-2xl p-6 border border-indigo-900/30">
//             <h2 className="text-white text-xl mb-4">Example Document Editor</h2>
//             <Editor docId={1} workspaceId={"default"} />
//           </div>
//         </div>
//       </div>
//     </div>
//   );
// }
import { FolderKanban, Plus } from 'lucide-react';
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getWorkspaces, createWorkspace } from "../../services/api";
import { showToast } from "../ui/useToast";
export function WorkspacesPage() {
  const [workspaces, setWorkspaces] = useState<any[]>([]);
  const [name, setName] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const load = async () => {
    try {
      setLoading(true);
      const data = await getWorkspaces();
      setWorkspaces(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleCreate = async () => {
    if (!name) return;
    try {
      await createWorkspace(name);
      setName("");
      showToast("Workspace created");
      load();
    } catch (err) {
      console.error(err);
      showToast("Failed to create workspace");
    }
  };

  useEffect(() => {
    load();
  }, []);

  return (
    <div className="flex-1 overflow-y-auto bg-gradient-to-br from-[#0a0e27] to-[#1a1f4d]">
      <div className="p-8 max-w-7xl mx-auto">

        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl text-white mb-2">Workspaces</h1>
            <p className="text-gray-400">Organize your research projects</p>
          </div>

          <div className="flex gap-2">
            <input
              className="px-3 py-2 rounded bg-white/10 text-white border border-indigo-500"
              placeholder="Workspace name"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />

            <button
              onClick={handleCreate}
              className="px-6 py-2 bg-gradient-to-r from-indigo-500 to-purple-600 text-white rounded-lg flex items-center gap-2"
            >
              <Plus className="w-5 h-5" />
              New
            </button>
          </div>
        </div>

        {/* Grid */}
        {loading && (
          <div className="text-gray-400 text-sm mb-4">Loading workspaces...</div>
        )}

        {!loading && workspaces.length === 0 && (
          <div className="text-gray-400 text-sm">No workspaces yet. Create your first workspace to get started.</div>
        )}

        <div className="grid grid-cols-2 gap-6 mt-4">
          {workspaces.map((workspace) => (
            <div
              key={workspace.id}
              onClick={() => navigate(`/workspace/${workspace.id}`)}
              className="bg-white/5 backdrop-blur-xl rounded-2xl p-6 border border-indigo-900/30 hover:bg-white/10 transition-all cursor-pointer group"
            >
              <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center mb-4">
                <FolderKanban className="w-6 h-6 text-white" />
              </div>

              <h3 className="text-xl text-white group-hover:text-indigo-300">
                {workspace.name}
              </h3>

              <p className="text-sm text-gray-400 mt-2">
                ID: {workspace.id}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}