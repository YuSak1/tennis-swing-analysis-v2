import { useState, useRef, useCallback, useMemo } from "react";
import { Upload, X, Film } from "lucide-react";

interface VideoUploaderProps {
  file: File | null;
  onFileChange: (file: File | null) => void;
}

export default function VideoUploader({ file, onFileChange }: VideoUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const previewUrl = useMemo(
    () => (file ? URL.createObjectURL(file) : null),
    [file]
  );

  const handleFile = useCallback(
    (f: File | undefined) => {
      if (f && f.type === "video/mp4") {
        onFileChange(f);
      }
    },
    [onFileChange]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragging(false);
      const f = e.dataTransfer.files[0];
      handleFile(f);
    },
    [handleFile]
  );

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => setIsDragging(false);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    handleFile(e.target.files?.[0]);
  };

  const clearFile = () => {
    onFileChange(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  if (previewUrl && file) {
    return (
      <div className="relative rounded-2xl overflow-hidden bg-surface-900 border border-surface-700">
        <video
          src={previewUrl}
          className="w-full max-h-80 object-contain bg-black"
          controls
          muted
        />
        <button
          onClick={clearFile}
          className="absolute top-3 right-3 w-8 h-8 rounded-full bg-surface-900/80 backdrop-blur-sm flex items-center justify-center hover:bg-red-600 transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
        <div className="px-4 py-3 flex items-center gap-2 text-sm text-surface-300">
          <Film className="w-4 h-4 text-court-400" />
          <span className="truncate">{file.name}</span>
          <span className="text-surface-500 ml-auto">
            {(file.size / 1024 / 1024).toFixed(1)} MB
          </span>
        </div>
      </div>
    );
  }

  return (
    <div
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onClick={() => inputRef.current?.click()}
      className={`
        relative rounded-2xl border-2 border-dashed cursor-pointer
        flex flex-col items-center justify-center gap-4 py-16 px-8
        transition-all duration-200
        ${
          isDragging
            ? "border-court-400 bg-court-400/5 scale-[1.01]"
            : "border-surface-700 hover:border-surface-500 bg-surface-900/50"
        }
      `}
    >
      <div
        className={`w-14 h-14 rounded-2xl flex items-center justify-center transition-colors ${
          isDragging ? "bg-court-600/20" : "bg-surface-800"
        }`}
      >
        <Upload
          className={`w-6 h-6 ${isDragging ? "text-court-400" : "text-surface-400"}`}
        />
      </div>
      <div className="text-center">
        <p className="text-surface-200 font-medium">
          Drop your video here or click to browse
        </p>
        <p className="text-surface-500 text-sm mt-1">MP4 format, 3–30 seconds</p>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="video/mp4"
        onChange={handleInputChange}
        className="hidden"
      />
    </div>
  );
}
