import { Sparkles, ArrowRight, Brain, FileSearch, Zap } from 'lucide-react';
import { Link } from 'react-router-dom';

export function LandingPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-[#0a0e27] via-[#1a1f4d] to-[#0a0e27] relative overflow-hidden">
      {/* AI Network Background Pattern */}
      <div className="absolute inset-0 opacity-20">
        <div className="absolute top-20 left-20 w-96 h-96 bg-indigo-500 rounded-full blur-3xl"></div>
        <div className="absolute bottom-20 right-20 w-96 h-96 bg-purple-500 rounded-full blur-3xl"></div>
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-blue-500 rounded-full blur-3xl"></div>
      </div>

      {/* Header */}
      <header className="relative z-10 p-6 flex items-center justify-between max-w-7xl mx-auto">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
            <Sparkles className="w-6 h-6 text-white" />
          </div>
          <div className="text-white font-semibold text-lg">ResearchPilot</div>
        </div>
        <Link
          to="/dashboard"
          className="px-4 py-2 text-sm text-indigo-300 hover:text-white transition-colors"
        >
          Sign In
        </Link>
      </header>

      {/* Hero Section */}
      <div className="relative z-10 max-w-5xl mx-auto px-6 pt-20 pb-32 text-center">
        <div className="inline-block mb-6 px-4 py-2 bg-indigo-500/20 rounded-full border border-indigo-500/30">
          <span className="text-indigo-300 text-sm">Autonomous Research Intelligence Hub</span>
        </div>

        <h1 className="text-6xl font-bold text-white mb-6 leading-tight">
          ResearchPilot AI Agent
        </h1>

        <p className="text-2xl text-gray-300 mb-12 max-w-3xl mx-auto">
          Accelerate Literature Review with Autonomous Intelligence
        </p>

        <Link
          to="/dashboard"
          className="inline-flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 text-white rounded-lg transition-all shadow-lg shadow-indigo-500/50 text-lg"
        >
          Start Research
          <ArrowRight className="w-5 h-5" />
        </Link>

        {/* Feature Cards */}
        <div className="grid grid-cols-3 gap-6 mt-20">
          <div className="bg-white/5 backdrop-blur-xl rounded-2xl p-6 border border-white/10">
            <div className="w-12 h-12 rounded-lg bg-indigo-500/20 flex items-center justify-center mb-4 mx-auto">
              <Brain className="w-6 h-6 text-indigo-400" />
            </div>
            <h3 className="text-white font-semibold mb-2">AI-Powered Analysis</h3>
            <p className="text-gray-400 text-sm">
              Advanced NLP algorithms analyze research papers instantly
            </p>
          </div>

          <div className="bg-white/5 backdrop-blur-xl rounded-2xl p-6 border border-white/10">
            <div className="w-12 h-12 rounded-lg bg-purple-500/20 flex items-center justify-center mb-4 mx-auto">
              <FileSearch className="w-6 h-6 text-purple-400" />
            </div>
            <h3 className="text-white font-semibold mb-2">Smart Document Search</h3>
            <p className="text-gray-400 text-sm">
              Find relevant papers across millions of publications
            </p>
          </div>

          <div className="bg-white/5 backdrop-blur-xl rounded-2xl p-6 border border-white/10">
            <div className="w-12 h-12 rounded-lg bg-blue-500/20 flex items-center justify-center mb-4 mx-auto">
              <Zap className="w-6 h-6 text-blue-400" />
            </div>
            <h3 className="text-white font-semibold mb-2">Instant Insights</h3>
            <p className="text-gray-400 text-sm">
              Generate comprehensive summaries in seconds
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
