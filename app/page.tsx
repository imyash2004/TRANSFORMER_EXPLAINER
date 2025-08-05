"use client"

import { useState, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Textarea } from "@/components/ui/textarea"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import {
  ChevronRight,
  ChevronLeft,
  Eye,
  Brain,
  Zap,
  AlertTriangle,
  CheckCircle,
  Moon,
  Sun,
  Info,
  Calculator,
  Network,
  Target,
  Lightbulb,
  Download,
  Sparkles,
} from "lucide-react"

// Import our modular components
import { TokenVisualization } from "@/components/TokenVisualization"
import { EmbeddingVisualization } from "@/components/EmbeddingVisualization"
import { AttentionVisualization } from "@/components/AttentionVisualization"
import { analyzeText, type AnalysisResult } from "@/lib/textAnalyzer"

// Enhanced examples with more variety for beginners
const ENHANCED_EXAMPLES = [
  {
    id: 1,
    text: "write a poem on winter",
    category: "Creative Writing",
    icon: "❄️",
    gradient: "from-blue-400 to-blue-600",
    description: "Generate creative poetry about winter themes",
    difficulty: "Beginner",
    explanation:
      "This is a creative writing task. The AI needs to understand 'write' as an instruction, 'poem' as the format, and 'winter' as the theme. It will use its knowledge of poetry structure and winter imagery to create original content.",
  },
  {
    id: 2,
    text: "what is the capital of France",
    category: "Factual Question",
    icon: "🗼",
    gradient: "from-green-400 to-green-600",
    description: "Simple factual question requiring specific knowledge",
    difficulty: "Beginner",
    explanation:
      "This is a straightforward factual question. The AI recognizes 'what is' as a question pattern, 'capital' as a geographic concept, and 'France' as a country. It retrieves the stored fact that Paris is France's capital.",
  },
  {
    id: 3,
    text: "explain machine learning to a 5 year old",
    category: "Educational",
    icon: "🧠",
    gradient: "from-purple-400 to-purple-600",
    description: "Complex concept simplified for young audience",
    difficulty: "Intermediate",
    explanation:
      "This requires the AI to understand multiple concepts: 'explain' (instruction), 'machine learning' (complex topic), and 'to a 5 year old' (audience specification). It must simplify technical concepts using analogies and simple language.",
  },
  {
    id: 4,
    text: "translate hello world to Spanish",
    category: "Translation",
    icon: "🌍",
    gradient: "from-orange-400 to-orange-600",
    description: "Language translation task",
    difficulty: "Beginner",
    explanation:
      "Translation task where the AI identifies 'translate' as the action, 'hello world' as the source text, and 'Spanish' as the target language. It uses its multilingual training to provide the equivalent phrase.",
  },
  {
    id: 5,
    text: "compare and contrast cats and dogs as pets",
    category: "Analysis",
    icon: "🐱",
    gradient: "from-red-400 to-red-600",
    description: "Analytical comparison requiring structured thinking",
    difficulty: "Advanced",
    explanation:
      "This complex task requires the AI to understand 'compare and contrast' as an analytical framework, identify 'cats and dogs' as subjects, and 'as pets' as the context. It must organize information into similarities and differences.",
  },
  {
    id: 6,
    text: "once upon a time in a magical forest",
    category: "Storytelling",
    icon: "🌲",
    gradient: "from-emerald-400 to-emerald-600",
    description: "Story beginning that triggers narrative generation",
    difficulty: "Intermediate",
    explanation:
      "This storytelling prompt uses the classic 'once upon a time' opening to signal narrative mode. The AI recognizes this pattern and prepares to generate a story with characters, setting, and plot development.",
  },
]

export default function ComprehensiveLLMWorkshop() {
  const [currentModule, setCurrentModule] = useState(1)
  const [selectedExample, setSelectedExample] = useState<number>(1)
  const [customInput, setCustomInput] = useState("")
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null)
  const [darkMode, setDarkMode] = useState(false)
  const [selectedToken, setSelectedToken] = useState<number | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentStep, setCurrentStep] = useState(0)
  const [showMath, setShowMath] = useState(false)

  const progress = (currentModule / 6) * 100

  // Auto-play functionality
  useEffect(() => {
    let interval: NodeJS.Timeout
    if (isPlaying && analysisResult) {
      interval = setInterval(() => {
        setCurrentStep((prev) => (prev + 1) % analysisResult.tokens.length)
      }, 2000)
    }
    return () => clearInterval(interval)
  }, [isPlaying, analysisResult])

  // Analyze selected example
  useEffect(() => {
    const example = ENHANCED_EXAMPLES.find((e) => e.id === selectedExample)
    if (example) {
      const result = analyzeText(example.text)
      setAnalysisResult(result)
    }
  }, [selectedExample])

  // Analyze custom input
  const handleCustomAnalysis = () => {
    if (customInput.trim()) {
      const result = analyzeText(customInput.trim())
      setAnalysisResult(result)
      setSelectedExample(0) // Clear example selection
    }
  }

  const MODULES = [
    { id: 1, name: "Input Selection", icon: Eye, description: "Choose and analyze input text" },
    { id: 2, name: "Tokenization", icon: Zap, description: "Break text into tokens" },
    { id: 3, name: "Embeddings 3D", icon: Brain, description: "Visualize semantic space" },
    { id: 4, name: "Attention", icon: Network, description: "Multi-head attention mechanism" },
    { id: 5, name: "Generation", icon: Target, description: "Token prediction and sampling" },
    { id: 6, name: "Limitations", icon: AlertTriangle, description: "Understanding model limitations" },
  ]

  const renderModuleContent = () => {
    switch (currentModule) {
      case 1:
        return (
          <div className="space-y-8">
            <div className="text-center">
              <h2 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-4">
                Input Selection & Analysis
              </h2>
              <p className="text-xl text-gray-600 dark:text-gray-300 max-w-4xl mx-auto">
                Choose your input text and explore how different types of prompts activate different model behaviors and
                capabilities.
              </p>
            </div>

            {/* Example Selection Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {ENHANCED_EXAMPLES.map((example) => (
                <motion.div key={example.id} whileHover={{ scale: 1.02, y: -5 }} whileTap={{ scale: 0.98 }}>
                  <Card
                    className={`cursor-pointer transition-all duration-300 ${
                      selectedExample === example.id ? "ring-2 ring-blue-500 shadow-xl" : "hover:shadow-lg"
                    }`}
                    onClick={() => setSelectedExample(example.id)}
                  >
                    <CardContent className="p-6">
                      <div
                        className={`w-full h-32 bg-gradient-to-br ${example.gradient} rounded-lg flex items-center justify-center mb-4 relative overflow-hidden`}
                      >
                        <span className="text-4xl z-10">{example.icon}</span>
                        <div className="absolute inset-0 bg-white/10 backdrop-blur-sm" />
                      </div>

                      <div className="space-y-3">
                        <div className="flex items-center justify-between">
                          <h3 className="font-semibold text-lg">{example.category}</h3>
                          <Badge
                            variant={
                              example.difficulty === "Beginner"
                                ? "default"
                                : example.difficulty === "Intermediate"
                                  ? "secondary"
                                  : "destructive"
                            }
                          >
                            {example.difficulty}
                          </Badge>
                        </div>

                        <p className="text-sm text-gray-600 dark:text-gray-300">{example.description}</p>

                        <div className="bg-gray-100 dark:bg-gray-700 rounded-lg p-3">
                          <code className="text-sm font-mono">{example.text}</code>
                        </div>

                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                          <p className="text-xs text-blue-700 dark:text-blue-300">{example.explanation}</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </div>

            {/* Custom Input */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Lightbulb className="h-5 w-5" />
                  <span>Custom Input Analysis</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <Textarea
                  placeholder="Enter your own text to analyze... Try different types: questions, creative prompts, technical queries, or casual conversation."
                  value={customInput}
                  onChange={(e) => setCustomInput(e.target.value)}
                  className="min-h-[100px] text-lg"
                />
                <div className="flex justify-between items-center">
                  <div className="text-sm text-gray-500">
                    {customInput.length}/500 characters • {customInput.split(" ").filter((w) => w).length} words
                  </div>
                  <Button onClick={handleCustomAnalysis} disabled={!customInput.trim()}>
                    <Sparkles className="h-4 w-4 mr-2" />
                    Analyze Text
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Analysis Results */}
            {analysisResult && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <span>Analysis Results</span>
                    <div className="flex items-center space-x-2">
                      <Badge variant="outline">{analysisResult.category}</Badge>
                      <Badge
                        variant={
                          analysisResult.complexity === "beginner"
                            ? "default"
                            : analysisResult.complexity === "intermediate"
                              ? "secondary"
                              : "destructive"
                        }
                      >
                        {analysisResult.complexity}
                      </Badge>
                      <Badge variant="outline">{(analysisResult.confidence * 100).toFixed(0)}% confidence</Badge>
                    </div>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                      <h4 className="font-medium mb-2">How the AI understands this input:</h4>
                      <p className="text-sm text-gray-700 dark:text-gray-300">{analysisResult.explanation}</p>
                    </div>

                    <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                      <h4 className="font-medium mb-2">Expected Response Type:</h4>
                      <p className="text-sm text-gray-700 dark:text-gray-300">{analysisResult.predictedResponse}</p>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="text-center p-3 bg-white dark:bg-gray-700 rounded-lg">
                        <div className="text-2xl font-bold text-blue-600">{analysisResult.tokens.length}</div>
                        <div className="text-sm text-gray-600">Tokens</div>
                      </div>
                      <div className="text-center p-3 bg-white dark:bg-gray-700 rounded-lg">
                        <div className="text-2xl font-bold text-green-600">
                          {new Set(analysisResult.tokens.map((t) => t.semantic_role)).size}
                        </div>
                        <div className="text-sm text-gray-600">Semantic Roles</div>
                      </div>
                      <div className="text-center p-3 bg-white dark:bg-gray-700 rounded-lg">
                        <div className="text-2xl font-bold text-purple-600">
                          {new Set(analysisResult.tokens.map((t) => t.pos_tag)).size}
                        </div>
                        <div className="text-sm text-gray-600">POS Tags</div>
                      </div>
                      <div className="text-center p-3 bg-white dark:bg-gray-700 rounded-lg">
                        <div className="text-2xl font-bold text-orange-600">
                          {(analysisResult.confidence * 100).toFixed(0)}%
                        </div>
                        <div className="text-sm text-gray-600">Confidence</div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        )

      case 2:
        return (
          <div className="space-y-8">
            <div className="text-center">
              <h2 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-4">
                Tokenization Process
              </h2>
              <p className="text-xl text-gray-600 dark:text-gray-300 max-w-4xl mx-auto">
                Watch how your input text is broken down into tokens - the fundamental units that the model processes.
                Think of tokens as the "words" that the AI actually sees and understands.
              </p>
            </div>

            {analysisResult && (
              <TokenVisualization
                tokens={analysisResult.tokens}
                selectedToken={selectedToken}
                onTokenSelect={setSelectedToken}
                currentStep={currentStep}
                isPlaying={isPlaying}
                onPlayToggle={() => setIsPlaying(!isPlaying)}
                onReset={() => setCurrentStep(0)}
              />
            )}

            {!analysisResult && (
              <Alert>
                <Info className="h-4 w-4" />
                <AlertDescription>
                  Please select an example or enter custom text in Module 1 to see the tokenization process.
                </AlertDescription>
              </Alert>
            )}
          </div>
        )

      case 3:
        return (
          <div className="space-y-8">
            <div className="text-center">
              <h2 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-4">
                3D Embedding Space
              </h2>
              <p className="text-xl text-gray-600 dark:text-gray-300 max-w-4xl mx-auto">
                Explore how tokens are positioned in high-dimensional space based on their semantic meaning. Words with
                similar meanings cluster together in this mathematical space.
              </p>
            </div>

            {analysisResult && (
              <EmbeddingVisualization
                tokens={analysisResult.tokens}
                selectedToken={selectedToken}
                onTokenSelect={setSelectedToken}
              />
            )}

            {!analysisResult && (
              <Alert>
                <Info className="h-4 w-4" />
                <AlertDescription>
                  Please select an example or enter custom text in Module 1 to see the embedding visualization.
                </AlertDescription>
              </Alert>
            )}
          </div>
        )

      case 4:
        return (
          <div className="space-y-8">
            <div className="text-center">
              <h2 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-4">
                Multi-Head Attention Mechanism
              </h2>
              <p className="text-xl text-gray-600 dark:text-gray-300 max-w-4xl mx-auto">
                Visualize how the model pays attention to different parts of the input when processing each token. Each
                attention head specializes in different types of relationships.
              </p>
            </div>

            {analysisResult && (
              <AttentionVisualization
                tokens={analysisResult.tokens}
                selectedToken={selectedToken}
                onTokenSelect={setSelectedToken}
              />
            )}

            {!analysisResult && (
              <Alert>
                <Info className="h-4 w-4" />
                <AlertDescription>
                  Please select an example or enter custom text in Module 1 to see the attention visualization.
                </AlertDescription>
              </Alert>
            )}
          </div>
        )

      case 5:
        return (
          <div className="space-y-8">
            <div className="text-center">
              <h2 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-4">
                Token Generation Process
              </h2>
              <p className="text-xl text-gray-600 dark:text-gray-300 max-w-4xl mx-auto">
                See how the model predicts the next token based on probability distributions and sampling parameters.
                This is where the magic of text generation happens!
              </p>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>Coming Soon: Generation Visualization</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center py-12">
                  <Target className="h-16 w-16 mx-auto text-gray-400 mb-4" />
                  <p className="text-gray-600 dark:text-gray-300">
                    Interactive generation visualization with probability distributions, temperature controls, and
                    real-time sampling will be available here.
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
        )

      case 6:
        return (
          <div className="space-y-8">
            <div className="text-center">
              <h2 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-4">
                Understanding Model Limitations
              </h2>
              <p className="text-xl text-gray-600 dark:text-gray-300 max-w-4xl mx-auto">
                Learn about the boundaries and limitations of Large Language Models to use them responsibly and
                effectively.
              </p>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>Coming Soon: Limitations Explorer</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center py-12">
                  <AlertTriangle className="h-16 w-16 mx-auto text-gray-400 mb-4" />
                  <p className="text-gray-600 dark:text-gray-300">
                    Interactive examples of model limitations, biases, and failure modes with explanations and
                    mitigation strategies will be available here.
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
        )

      default:
        return <div>Module {currentModule} content...</div>
    }
  }

  return (
    <TooltipProvider>
      <div className={`min-h-screen ${darkMode ? "dark bg-gray-900" : "bg-gray-50"}`}>
        {/* Enhanced Header */}
        <header className="bg-white dark:bg-gray-800 shadow-sm border-b sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center h-16">
              <div className="flex items-center space-x-4">
                <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  TRANSFORMER EXPLAINER
                </h1>
                <Badge variant="secondary">Interactive Workshop</Badge>
              </div>

              <div className="flex items-center space-x-4">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowMath(!showMath)}
                  className="flex items-center space-x-2"
                >
                  <Calculator className="h-4 w-4" />
                  <span>Math</span>
                </Button>

                <Button variant="ghost" size="icon" onClick={() => setDarkMode(!darkMode)}>
                  {darkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
                </Button>

                <Button variant="outline" size="sm">
                  <Download className="h-4 w-4 mr-2" />
                  Export
                </Button>
              </div>
            </div>
          </div>
          <Progress value={progress} className="h-1" />
        </header>

        <div className="flex">
          {/* Enhanced Sidebar */}
          <aside className="w-80 bg-white dark:bg-gray-800 shadow-sm h-screen sticky top-16 overflow-y-auto">
            <div className="p-6">
              <div className="mb-6">
                <h3 className="font-semibold text-lg mb-2">Learning Modules</h3>
                <p className="text-sm text-gray-600 dark:text-gray-300">
                  Complete interactive journey through LLM architecture
                </p>
              </div>

              <nav className="space-y-3">
                {MODULES.map((module) => {
                  const Icon = module.icon
                  return (
                    <TooltipProvider key={module.id}>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <button
                            onClick={() => setCurrentModule(module.id)}
                            className={`w-full flex items-start space-x-3 px-4 py-3 rounded-lg text-left transition-all ${
                              currentModule === module.id
                                ? "bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 shadow-sm"
                                : "text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                            }`}
                          >
                            <Icon className="h-5 w-5 mt-0.5 flex-shrink-0" />
                            <div className="flex-1 min-w-0">
                              <div className="font-medium text-sm">{module.name}</div>
                              <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">{module.description}</div>
                            </div>
                            {currentModule === module.id && (
                              <CheckCircle className="h-4 w-4 text-green-500 flex-shrink-0" />
                            )}
                          </button>
                        </TooltipTrigger>
                        <TooltipContent side="right">
                          <p>{module.description}</p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  )
                })}
              </nav>

              {/* Progress Indicator */}
              <div className="mt-6 pt-6 border-t">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium">Progress</span>
                  <span className="text-sm text-gray-500">{currentModule}/6</span>
                </div>
                <div className="flex space-x-1">
                  {MODULES.map((_, index) => (
                    <div
                      key={index}
                      className={`flex-1 h-2 rounded ${
                        index + 1 <= currentModule ? "bg-blue-500" : "bg-gray-200 dark:bg-gray-700"
                      }`}
                    />
                  ))}
                </div>
              </div>

              {/* Quick Stats */}
              {analysisResult && (
                <div className="mt-6 pt-6 border-t">
                  <h4 className="font-medium text-sm mb-3">Current Analysis</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between text-xs">
                      <span>Tokens:</span>
                      <span className="font-mono">{analysisResult.tokens.length}</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span>Category:</span>
                      <span className="font-mono">{analysisResult.category}</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span>Complexity:</span>
                      <span className="font-mono capitalize">{analysisResult.complexity}</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span>Confidence:</span>
                      <span className="font-mono">{(analysisResult.confidence * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </aside>

          {/* Main Content */}
          <main className="flex-1 p-8">
            <AnimatePresence mode="wait">
              <motion.div
                key={currentModule}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3 }}
              >
                {renderModuleContent()}
              </motion.div>
            </AnimatePresence>

            {/* Enhanced Navigation */}
            <div className="flex justify-between items-center mt-12 pt-8 border-t">
              <Button
                variant="outline"
                onClick={() => setCurrentModule(Math.max(1, currentModule - 1))}
                disabled={currentModule === 1}
                className="flex items-center space-x-2"
              >
                <ChevronLeft className="h-4 w-4" />
                <span>Previous Module</span>
              </Button>

              <div className="flex items-center space-x-4">
                <div className="text-sm text-gray-500">Module {currentModule} of 6</div>
                <div className="flex space-x-1">
                  {MODULES.map((_, index) => (
                    <button
                      key={index}
                      onClick={() => setCurrentModule(index + 1)}
                      className={`w-3 h-3 rounded-full transition-colors ${
                        index + 1 === currentModule
                          ? "bg-blue-500"
                          : index + 1 < currentModule
                            ? "bg-green-500"
                            : "bg-gray-300 hover:bg-gray-400"
                      }`}
                    />
                  ))}
                </div>
              </div>

              <Button
                onClick={() => setCurrentModule(Math.min(6, currentModule + 1))}
                disabled={currentModule === 6}
                className="flex items-center space-x-2"
              >
                <span>Next Module</span>
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          </main>
        </div>
      </div>
    </TooltipProvider>
  )
}
