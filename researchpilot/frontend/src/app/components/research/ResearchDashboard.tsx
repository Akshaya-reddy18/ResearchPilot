import { Search, Sparkles, TrendingUp, Lightbulb, GitBranch, ArrowUpRight, Loader2 } from 'lucide-react';
import { useState } from 'react';
import { chat, type ChatResponse } from '../../services/api';

interface ResearchReport {
  executiveSummary: string;
  keyFindings: string[];
  methodologyComparison: string;
  researchGaps: string[];
  futureDirections: string[];
}

export function ResearchDashboard() {
  const [topic, setTopic] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [report, setReport] = useState<ResearchReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  const parseAnalysis = (analysis: string): ResearchReport => {
    const sections: ResearchReport = {
      executiveSummary: '',
      keyFindings: [],
      methodologyComparison: '',
      researchGaps: [],
      futureDirections: [],
    };

    // Parse structured response
    const lines = analysis.split('\n');
    let currentSection: keyof ResearchReport | null = null;

    for (const line of lines) {
      const trimmed = line.trim();
      
      // Match section headers more flexibly
      if (trimmed.match(/^1[\.\)]\s*Executive Summary/i)) {
        currentSection = 'executiveSummary';
        continue; // Skip the header line itself
      } else if (trimmed.match(/^2[\.\)]\s*Key Findings/i)) {
        currentSection = 'keyFindings';
        continue;
      } else if (trimmed.match(/^3[\.\)]\s*Methodology Comparison/i)) {
        currentSection = 'methodologyComparison';
        continue;
      } else if (trimmed.match(/^4[\.\)]\s*Research Gaps/i)) {
        currentSection = 'researchGaps';
        continue;
      } else if (trimmed.match(/^5[\.\)]\s*Future Scope/i) || trimmed.match(/^5[\.\)]\s*Future Directions/i)) {
        currentSection = 'futureDirections';
        continue;
      }
      
      // Process content for current section
      if (currentSection && trimmed && !trimmed.match(/^\d+[\.\)]/)) {
        if (currentSection === 'keyFindings' || currentSection === 'researchGaps' || currentSection === 'futureDirections') {
          // For list sections, accept various formats
          if (trimmed.startsWith('-') || trimmed.startsWith('•') || trimmed.startsWith('*')) {
            sections[currentSection].push(trimmed.substring(1).trim());
          } else if (trimmed.length > 0 && !trimmed.match(/^[A-Z][a-z]+\s*:/)) {
            // Add as list item if it's not a header
            sections[currentSection].push(trimmed);
          }
        } else {
          // For text sections, append content
          if (trimmed.length > 0) {
            sections[currentSection] += (sections[currentSection] ? '\n' : '') + trimmed;
          }
        }
      }
    }

    return sections;
  };

  const handleGenerate = async () => {
    if (!topic.trim()) {
      setError('Please enter a research topic');
      return;
    }

    setIsGenerating(true);
    setError(null);
    setReport(null);

    try {
      console.log('Generating report for topic:', topic);
      const response: ChatResponse = await chat({
        query: `Generate a comprehensive research analysis on: ${topic}`,
        use_context: true,
        top_k: 10,
      });
      console.log('Report response:', response);

      if (response.analysis) {
        const parsed = parseAnalysis(response.analysis);
        // Check if parsing found any content
        const hasContent = parsed.executiveSummary || 
                          parsed.keyFindings.length > 0 || 
                          parsed.methodologyComparison || 
                          parsed.researchGaps.length > 0 || 
                          parsed.futureDirections.length > 0;
        
        if (hasContent) {
          setReport(parsed);
        } else {
          // Parsing failed, show raw analysis as fallback
          console.warn('Parsing failed, showing raw analysis');
          setReport({
            executiveSummary: response.analysis,
            keyFindings: [],
            methodologyComparison: '',
            researchGaps: [],
            futureDirections: [],
          });
        }
      } else {
        setError('Failed to generate analysis. Please try again.');
      }
    } catch (err) {
      console.error('Report generation error:', err);
      const errorMessage = err instanceof Error ? err.message : 'Failed to generate report';
      setError(errorMessage);
      // Clear any partial report on error
      setReport(null);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="flex-1 overflow-y-auto bg-gradient-to-br from-[#0a0e27] to-[#1a1f4d]">
      <div className="p-8 max-w-7xl mx-auto">
        {/* Search Section */}
        <div className="mb-8">
          <h1 className="text-3xl text-white mb-6">Research Dashboard</h1>
          
          <div className="flex gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                placeholder="Enter Research Topic…"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                className="w-full pl-12 pr-4 py-4 bg-white/5 border border-indigo-900/30 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-indigo-500/50 focus:bg-white/10 transition-all"
              />
            </div>
            <button
              onClick={handleGenerate}
              disabled={isGenerating}
              className="px-8 py-4 bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg transition-all shadow-lg shadow-indigo-500/30 flex items-center gap-2"
            >
              {isGenerating ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Sparkles className="w-5 h-5" />
                  Generate Report
                </>
              )}
            </button>
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-red-500/20 border border-red-500/30 rounded-2xl p-4 mb-6">
            <p className="text-red-300">{error}</p>
          </div>
        )}

        {/* Content Sections */}
        {report ? (
          <div className="space-y-6">
            {/* Executive Summary */}
            {report.executiveSummary && (
              <div className="bg-white/5 backdrop-blur-xl rounded-2xl p-6 border border-indigo-900/30">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 rounded-lg bg-indigo-500/20 flex items-center justify-center">
                    <Sparkles className="w-5 h-5 text-indigo-400" />
                  </div>
                  <h2 className="text-xl text-white">Executive Summary</h2>
                </div>
                <p className="text-gray-300 leading-relaxed whitespace-pre-line">
                  {report.executiveSummary || 'No summary available'}
                </p>
              </div>
            )}

            {/* Key Findings */}
            {report.keyFindings.length > 0 && (
              <div className="bg-white/5 backdrop-blur-xl rounded-2xl p-6 border border-indigo-900/30">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 rounded-lg bg-purple-500/20 flex items-center justify-center">
                    <TrendingUp className="w-5 h-5 text-purple-400" />
                  </div>
                  <h2 className="text-xl text-white">Key Findings</h2>
                </div>
                <div className="space-y-3">
                  {report.keyFindings.map((finding, index) => (
                    <div key={index} className="flex items-start gap-3 p-3 bg-white/5 rounded-lg border border-white/10">
                      <div className="w-6 h-6 rounded-full bg-indigo-500/20 flex items-center justify-center shrink-0 mt-0.5">
                        <span className="text-xs text-indigo-400">{index + 1}</span>
                      </div>
                      <p className="text-gray-300 text-sm">{finding}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Two Column Layout */}
            <div className="grid grid-cols-2 gap-6">
              {/* Methodology Comparison */}
              {report.methodologyComparison && (
                <div className="bg-white/5 backdrop-blur-xl rounded-2xl p-6 border border-indigo-900/30">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="w-10 h-10 rounded-lg bg-blue-500/20 flex items-center justify-center">
                      <GitBranch className="w-5 h-5 text-blue-400" />
                    </div>
                    <h2 className="text-lg text-white">Methodology Comparison</h2>
                  </div>
                  <p className="text-gray-300 text-sm whitespace-pre-line">
                    {report.methodologyComparison}
                  </p>
                </div>
              )}

              {/* Research Gaps */}
              {report.researchGaps.length > 0 && (
                <div className="bg-white/5 backdrop-blur-xl rounded-2xl p-6 border border-indigo-900/30">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="w-10 h-10 rounded-lg bg-orange-500/20 flex items-center justify-center">
                      <Lightbulb className="w-5 h-5 text-orange-400" />
                    </div>
                    <h2 className="text-lg text-white">Research Gaps</h2>
                  </div>
                  <div className="space-y-3">
                    {report.researchGaps.map((gap, index) => (
                      <div key={index} className="flex items-start gap-2 text-sm text-gray-300">
                        <ArrowUpRight className="w-4 h-4 text-orange-400 shrink-0 mt-0.5" />
                        <span>{gap}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Future Directions */}
            {report.futureDirections.length > 0 && (
              <div className="bg-white/5 backdrop-blur-xl rounded-2xl p-6 border border-indigo-900/30">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 rounded-lg bg-green-500/20 flex items-center justify-center">
                    <ArrowUpRight className="w-5 h-5 text-green-400" />
                  </div>
                  <h2 className="text-xl text-white">Future Directions</h2>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  {report.futureDirections.map((direction, index) => (
                    <div key={index} className="p-4 bg-white/5 rounded-lg border border-white/10">
                      <p className="text-gray-300 text-sm">{direction}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="bg-white/5 backdrop-blur-xl rounded-2xl p-12 border border-indigo-900/30 text-center">
            <Sparkles className="w-16 h-16 text-indigo-400 mx-auto mb-4 opacity-50" />
            <p className="text-gray-400">
              Enter a research topic above and click "Generate Report" to get started.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
