import { Upload, File, CheckCircle, Clock, X, Loader2 } from 'lucide-react';
import { useState, useRef, useEffect } from 'react';
import { uploadPDF, ingestDocuments, getCollectionStats, type IngestionResponse } from '../../services/api';

interface Document {
  id: string;
  name: string;
  size: string;
  status: 'uploading' | 'processing' | 'indexed' | 'error';
  uploadedAt: string;
  error?: string;
}

export function UploadPDF() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isIngesting, setIsIngesting] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    loadDocumentStats();
  }, []);

  const loadDocumentStats = async () => {
    try {
      const stats = await getCollectionStats();
      // You could load document list from stats if available
    } catch (error) {
      console.error('Failed to load document stats:', error);
    }
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files).filter(file => file.type === 'application/pdf');
    if (files.length > 0) {
      await handleFiles(files);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFiles(Array.from(files));
    }
  };

  const handleFiles = async (files: File[]) => {
    for (const file of files) {
      if (file.type !== 'application/pdf') {
        continue;
      }

      const docId = `${Date.now()}-${Math.random()}`;
      const fileSize = (file.size / (1024 * 1024)).toFixed(2) + ' MB';

      const newDoc: Document = {
        id: docId,
        name: file.name,
        size: fileSize,
        status: 'uploading',
        uploadedAt: 'Just now',
      };

      setDocuments((prev) => [...prev, newDoc]);
      setIsUploading(true);

      try {
        console.log('Uploading file:', file.name);
        const response: IngestionResponse = await uploadPDF(file);
        console.log('Upload response:', response);
        
        setDocuments((prev) =>
          prev.map((doc) =>
            doc.id === docId
              ? {
                  ...doc,
                  status: response.status === 'success' ? 'indexed' : 'error',
                  error: response.status !== 'success' ? response.message : undefined,
                }
              : doc
          )
        );

        await loadDocumentStats();
      } catch (error) {
        console.error('Upload error:', error);
        setDocuments((prev) =>
          prev.map((doc) =>
            doc.id === docId
              ? {
                  ...doc,
                  status: 'error',
                  error: error instanceof Error ? error.message : 'Upload failed. Check console for details.',
                }
              : doc
          )
        );
      } finally {
        setIsUploading(false);
      }
    }
  };

  const handleIngestAll = async () => {
    setIsIngesting(true);
    try {
      const response: IngestionResponse = await ingestDocuments();
      if (response.status === 'success') {
        await loadDocumentStats();
        // Refresh document list or show success message
      }
    } catch (error) {
      console.error('Ingestion failed:', error);
    } finally {
      setIsIngesting(false);
    }
  };

  const removeDocument = (id: string) => {
    setDocuments((prev) => prev.filter((doc) => doc.id !== id));
  };

  return (
    <div className="flex-1 overflow-y-auto bg-gradient-to-br from-[#0a0e27] to-[#1a1f4d]">
      <div className="p-8 max-w-6xl mx-auto">
        <h1 className="text-3xl text-white mb-2">Upload Research Papers</h1>
        <p className="text-gray-400 mb-8">Upload PDF documents to build your research library</p>

        {/* Drag and Drop Upload Box */}
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          className="mb-8 border-2 border-dashed border-indigo-900/50 rounded-2xl bg-white/5 backdrop-blur-xl p-12 text-center hover:border-indigo-500/50 transition-all cursor-pointer"
        >
          <div className="w-16 h-16 rounded-full bg-indigo-500/20 flex items-center justify-center mx-auto mb-4">
            <Upload className="w-8 h-8 text-indigo-400" />
          </div>
          <h3 className="text-white text-lg mb-2">Drag and drop PDF files here</h3>
          <p className="text-gray-400 mb-4">or click to browse</p>
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf"
            multiple
            onChange={handleFileSelect}
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            className="px-6 py-2 bg-indigo-500/20 hover:bg-indigo-500/30 text-indigo-300 rounded-lg transition-all border border-indigo-500/30"
          >
            Choose Files
          </button>
          <p className="text-xs text-gray-500 mt-4">Supported format: PDF (max 10MB per file)</p>
        </div>

        {/* Uploaded Documents List */}
        <div className="bg-white/5 backdrop-blur-xl rounded-2xl p-6 border border-indigo-900/30">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl text-white">Uploaded Documents</h2>
            <div className="flex items-center gap-4">
              <span className="text-sm text-gray-400">{documents.length} documents</span>
              <button
                onClick={handleIngestAll}
                disabled={isIngesting || documents.length === 0}
                className="px-4 py-2 bg-indigo-500/20 hover:bg-indigo-500/30 disabled:opacity-50 disabled:cursor-not-allowed text-indigo-300 rounded-lg transition-all border border-indigo-500/30 text-sm flex items-center gap-2"
              >
                {isIngesting ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Ingesting...
                  </>
                ) : (
                  'Ingest All from Data Folder'
                )}
              </button>
            </div>
          </div>

          <div className="space-y-3">
            {documents.length === 0 ? (
              <p className="text-sm text-gray-500 italic text-center py-8">
                No documents uploaded yet. Drag and drop PDF files or click "Choose Files" to upload.
              </p>
            ) : (
              documents.map((doc) => (
              <div
                key={doc.id}
                className="flex items-center gap-4 p-4 bg-white/5 rounded-lg border border-white/10 hover:bg-white/10 transition-all"
              >
                {/* Icon */}
                <div className="w-12 h-12 rounded-lg bg-indigo-500/20 flex items-center justify-center shrink-0">
                  <File className="w-6 h-6 text-indigo-400" />
                </div>

                {/* Document Info */}
                <div className="flex-1 min-w-0">
                  <h3 className="text-white text-sm mb-1 truncate">{doc.name}</h3>
                  <p className="text-xs text-gray-400">
                    {doc.size} • {doc.uploadedAt}
                  </p>
                </div>

                {/* Status */}
                <div className="flex items-center gap-2">
                  {doc.status === 'indexed' ? (
                    <>
                      <CheckCircle className="w-5 h-5 text-green-400" />
                      <span className="text-sm text-green-400">Indexed</span>
                    </>
                  ) : doc.status === 'uploading' || doc.status === 'processing' ? (
                    <>
                      <Loader2 className="w-5 h-5 text-orange-400 animate-spin" />
                      <span className="text-sm text-orange-400">
                        {doc.status === 'uploading' ? 'Uploading...' : 'Processing...'}
                      </span>
                    </>
                  ) : doc.status === 'error' ? (
                    <>
                      <X className="w-5 h-5 text-red-400" />
                      <span className="text-sm text-red-400" title={doc.error}>
                        Error
                      </span>
                    </>
                  ) : null}
                </div>

                {/* Remove Button */}
                <button
                  onClick={() => removeDocument(doc.id)}
                  className="p-2 hover:bg-white/10 rounded-lg transition-all"
                >
                  <X className="w-4 h-4 text-gray-400 hover:text-red-400" />
                </button>
              </div>
            )))}
          </div>
        </div>
      </div>
    </div>
  );
}
