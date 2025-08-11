"use client"

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card"
import { Button } from "./ui/button"
import { Badge } from "./ui/badge"
import { Progress } from "./ui/progress"
import { Alert, AlertDescription } from "./ui/alert"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs"
import { Input } from "./ui/input"
import { Label } from "./ui/label"
import { Slider } from "./ui/slider"
import { AlertCircle, Brain, Calculator, Layers, Zap, Play, Pause, RotateCcw, ArrowRight, Edit3, Sparkles } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"

interface ProcessingStep {
  id: string
  name: string
  description: string
  status: 'pending' | 'processing' | 'complete'
  duration: number
  details: string[]
  technicalDetails?: string[]
  examples?: string[]
}

interface TextAnalysis {
  genre: string
  intent: string
  expectation: string
  confidence: number
  tokens: string[]
  complexity: number
  sentiment: string
}

const EXAMPLE_INPUTS = [
  {
    text: "Once upon a time in a magical forest",
    category: "Story Beginning",
    description: "Classic fairy tale opening"
  },
  {
    text: "Explain quantum physics in simple terms",
    category: "Educational Request", 
    description: "Science explanation query"
  },
  {
    text: "Write a Python function to sort a list",
    category: "Code Generation",
    description: "Programming task"
  },
  {
    text: "What are the main causes of climate change?",
    category: "Factual Question",
    description: "Knowledge retrieval"
  },
  {
    text: "I'm feeling overwhelmed with work lately",
    category: "Personal Conversation",
    description: "Emotional support context"
  }
]

export default function LLMResponseGeneratorEnhanced() {
  const [inputText, setInputText] = useState("Once upon a time in a magical forest")
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentStep, setCurrentStep] = useState(0)
  const [processedSteps, setProcessedSteps] = useState<string[]>([])
  const [showPrediction, setShowPrediction] = useState(false)
  const [animationSpeed, setAnimationSpeed] = useState([1])
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false)
  const [analysis, setAnalysis] = useState<TextAnalysis | null>(null)
  const [processingSteps, setProcessingSteps] = useState<ProcessingStep[]>([])

  // Analyze input text and generate processing steps
  const analyzeText = (text: string): TextAnalysis => {
    const tokens = text.toLowerCase().split(/\s+/)
    
    // Simple heuristics for demo purposes
    const genreDetection = () => {
      if (text.includes("once upon") || text.includes("magical") || text.includes("fairy")) return "Fantasy/Story"
      if (text.includes("explain") || text.includes("what is") || text.includes("how")) return "Educational"
      if (text.includes("write") || text.includes("function") || text.includes("code")) return "Code Generation"
      if (text.includes("feeling") || text.includes("I'm") || text.includes("help me")) return "Conversational"
      return "General Query"
    }

    const intentDetection = () => {
      if (text.includes("once upon") || text.includes("story")) return "Creative Writing"
      if (text.includes("explain") || text.includes("what")) return "Information Seeking"
      if (text.includes("write") || text.includes("create")) return "Content Generation"
      if (text.includes("help") || text.includes("how")) return "Problem Solving"
      return "General Interaction"
    }

    return {
      genre: genreDetection(),
      intent: intentDetection(),
      expectation: getExpectation(genreDetection(), intentDetection()),
      confidence: Math.random() * 0.3 + 0.7, // 70-100%
      tokens: tokens,
      complexity: Math.min(tokens.length / 10, 1),
      sentiment: detectSentiment(text)
    }
  }

  const getExpectation = (genre: string, intent: string): string => {
    if (genre === "Fantasy/Story") return "Narrative continuation with characters and plot development"
    if (genre === "Educational") return "Clear, structured explanation with examples"
    if (genre === "Code Generation") return "Functional code with comments and examples"
    if (genre === "Conversational") return "Empathetic, helpful response with suggestions"
    return "Contextually appropriate response based on query type"
  }

  const detectSentiment = (text: string): string => {
    const positiveWords = ["magical", "wonderful", "great", "amazing", "love"]
    const negativeWords = ["problem", "error", "difficult", "overwhelmed", "frustrated"]
    
    const positive = positiveWords.some(word => text.toLowerCase().includes(word))
    const negative = negativeWords.some(word => text.toLowerCase().includes(word))
    
    if (positive && !negative) return "Positive"
    if (negative && !positive) return "Negative"
    if (positive && negative) return "Mixed"
    return "Neutral"
  }

  const generateProcessingSteps = (text: string, analysis: TextAnalysis): ProcessingStep[] => {
    const tokens = analysis.tokens
    
    return [
      {
        id: "tokenization",
        name: "1. Tokenization & Preprocessing",
        description: "Breaking input into manageable pieces",
        status: 'pending' as const,
        duration: 500 / animationSpeed[0],
        details: [
          `📝 Input: "${text}"`,
          `🔤 Tokens: [${tokens.slice(0, 8).map(t => `'${t}'`).join(', ')}${tokens.length > 8 ? '...' : ''}]`,
          `📊 Token count: ${tokens.length} tokens`,
          `🏷️ Special tokens: [BOS] (Beginning of Sequence) added`,
          `🔍 Subword tokenization used for unknown words`
        ],
        technicalDetails: [
          "Byte-Pair Encoding (BPE) tokenization method",
          "Vocabulary size: ~50,000 unique tokens",
          "Maximum context length: 8,192 tokens",
          "Each token mapped to unique integer ID"
        ],
        examples: [
          "Common word: 'the' → token_id: 279",
          "Rare word: 'magical' → ['mag', 'ical'] (subwords)",
          "Punctuation: '.' → token_id: 13"
        ]
      },
      {
        id: "embeddings",
        name: "2. Token Embeddings & Positional Encoding",
        description: "Converting tokens to high-dimensional vectors",
        status: 'pending' as const,
        duration: 800 / animationSpeed[0],
        details: [
          `🎯 Each token → 4096-dimensional vector`,
          `📍 Position encoding added to preserve word order`,
          `🧮 '${tokens[0] || 'first'}' → [0.23, -0.15, 0.87, ...] (simplified)`,
          `⚡ Parallel processing for all ${tokens.length} tokens`,
          `🔗 Learned embeddings from training on trillions of tokens`
        ],
        technicalDetails: [
          "Embedding dimension: d_model = 4096",
          "Sinusoidal positional encoding used",
          "Embeddings are learned parameters (4096 × vocab_size)",
          "Position encodings ensure order sensitivity"
        ],
        examples: [
          "Word similarity: 'king' and 'queen' have similar embeddings",
          "Position matters: 'cat dog' ≠ 'dog cat' after encoding",
          "Rich representations capture syntax and semantics"
        ]
      },
      {
        id: "attention",
        name: "3. Multi-Head Self-Attention",
        description: "Understanding relationships between all words",
        status: 'pending' as const,
        duration: 1200 / animationSpeed[0],
        details: [
          `🧠 96 attention heads working in parallel`,
          `🔍 Each head specializes in different relationships`,
          `⚡ Head 1: Grammar patterns (${getGrammarExample(text)})`,
          `🎭 Head 2: Semantic meaning (${getSemanticExample(text)})`,
          `📍 Head 3: Position and word order relationships`,
          `🔗 Attention scores computed for all ${tokens.length}×${tokens.length} token pairs`
        ],
        technicalDetails: [
          "Scaled dot-product attention: softmax(QK^T/√d_k)V",
          "Query, Key, Value matrices learned during training",
          "Each head gets d_k = d_model/num_heads = 64 dimensions",
          "Attention provides interpretable token relationships"
        ],
        examples: [
          "Syntactic: 'Once' attends strongly to 'upon' (phrase pattern)",
          "Semantic: 'magical' attends to 'forest' (setting relationship)",
          "Long-range: First word connects to story theme words"
        ]
      },
      {
        id: "feedforward",
        name: "4. Feed-Forward Processing",
        description: "Deep feature extraction and refinement",
        status: 'pending' as const,
        duration: 900 / animationSpeed[0],
        details: [
          `🔄 Each token passes through feed-forward networks`,
          `📈 Dimension expansion: 4096 → 16384 → 4096`,
          `🧮 ReLU activation introduces non-linearity`,
          `🎯 Networks act as token-specific feature detectors`,
          `🔁 Process repeated across 96 transformer layers`
        ],
        technicalDetails: [
          "FFN(x) = max(0, xW₁ + b₁)W₂ + b₂",
          "Intermediate dimension: 4 × d_model = 16,384",
          "Residual connections prevent gradient vanishing",
          "Layer normalization for stable training"
        ],
        examples: [
          "Layer 10: Basic syntax understanding",
          "Layer 50: Complex semantic relationships", 
          "Layer 90: Task-specific representations"
        ]
      },
      {
        id: "context",
        name: "5. Contextual Understanding",
        description: "Building comprehensive meaning from all layers",
        status: 'pending' as const,
        duration: 1000 / animationSpeed[0],
        details: [
          `🎭 Genre detected: ${analysis.genre} (${(analysis.confidence * 100).toFixed(1)}% confidence)`,
          `🎯 Intent recognized: ${analysis.intent}`,
          `💭 Expected response: ${analysis.expectation}`,
          `😊 Sentiment analysis: ${analysis.sentiment}`,
          `🧩 Context assembled from 96 layers of processing`
        ],
        technicalDetails: [
          "Context vector represents entire input meaning",
          "Hierarchical feature extraction through layers",
          "Global coherence maintained across sequence",
          "Style and tone analysis integrated"
        ],
        examples: [
          "Story opening → Continue narrative",
          "Question format → Provide informative answer",
          "Code request → Generate functional code"
        ]
      },
      {
        id: "prediction",
        name: "6. Next Token Prediction",
        description: "Computing probabilities for all possible next words",
        status: 'pending' as const,
        duration: 900 / animationSpeed[0],
        details: [
          `📊 Probability distribution over 50,000+ vocabulary`,
          `🏆 Top predictions: ${getTopPredictions(text, analysis)}`,
          `🎲 Temperature controls randomness (currently 0.7)`,
          `⚡ Softmax ensures probabilities sum to 1.0`,
          `🧮 Based on complete contextual understanding`
        ],
        technicalDetails: [
          "Linear projection: hidden_state → vocab_logits",
          "Softmax normalization: exp(logit_i) / Σexp(logit_j)",
          "Temperature scaling affects distribution sharpness",
          "Top-k and nucleus sampling for variety"
        ],
        examples: [
          "Greedy: Always pick highest probability",
          "Sampling: Choose randomly from top candidates",
          "Beam search: Consider multiple paths"
        ]
      },
      {
        id: "generation",
        name: "7. Response Generation",
        description: "Iteratively building the complete response",
        status: 'pending' as const,
        duration: 1500 / animationSpeed[0],
        details: [
          `🔄 Autoregressive generation: one token at a time`,
          `➕ Each new token added to context for next prediction`,
          `🎯 Maintains consistency with original ${analysis.genre.toLowerCase()} style`,
          `⏹️ Stops at natural end or maximum length`,
          `📝 Full response: "${generateResponse(text, analysis)}"`
        ],
        technicalDetails: [
          "Context window slides with each new token",
          "Attention recomputed including new tokens",
          "Stopping criteria: [EOS] token or max_length",
          "Parallel generation possible with speculative decoding"
        ],
        examples: [
          "Story: Characters, plot, dialogue generation",
          "Code: Syntax highlighting, documentation",
          "Explanation: Step-by-step reasoning"
        ]
      }
    ]
  }

  const getGrammarExample = (text: string): string => {
    if (text.includes("once upon")) return "story opening pattern detected"
    if (text.includes("what") || text.includes("how")) return "question structure identified"
    if (text.includes("write") || text.includes("create")) return "imperative command recognized"
    return "syntactic relationships analyzed"
  }

  const getSemanticExample = (text: string): string => {
    if (text.includes("magical") && text.includes("forest")) return "fantasy setting relationship"
    if (text.includes("quantum") && text.includes("physics")) return "scientific domain connection"
    if (text.includes("Python") && text.includes("function")) return "programming context"
    return "semantic associations identified"
  }

  const getTopPredictions = (text: string, analysis: TextAnalysis): string => {
    if (analysis.genre === "Fantasy/Story") {
      return "• 'there' (24%), • 'lived' (19%), • 'where' (15%), • 'a' (13%)"
    } else if (analysis.genre === "Educational") {
      return "• 'is' (28%), • 'refers' (18%), • 'involves' (14%), • 'means' (12%)"
    } else if (analysis.genre === "Code Generation") {
      return "• 'def' (35%), • 'function' (22%), • 'that' (16%), • 'to' (11%)"
    } else {
      return "• 'is' (22%), • 'can' (18%), • 'would' (15%), • 'the' (13%)"
    }
  }

  const generateResponse = (text: string, analysis: TextAnalysis): string => {
    if (analysis.genre === "Fantasy/Story") {
      return text + " there lived a wise old owl named Luna, who possessed ancient magic and watched over all the woodland creatures. One day, a young adventurer stumbled into her domain..."
    } else if (analysis.genre === "Educational") {
      return "Quantum physics is the branch of physics that studies matter and energy at the smallest scales..."
    } else if (analysis.genre === "Code Generation") {
      return "```python\ndef sort_list(items):\n    return sorted(items)\n```"
    } else {
      return "I'd be happy to help you with that. Let me provide a comprehensive response..."
    }
  }

  // Effects and handlers
  useEffect(() => {
    const newAnalysis = analyzeText(inputText)
    setAnalysis(newAnalysis)
    setProcessingSteps(generateProcessingSteps(inputText, newAnalysis))
    resetDemo()
  }, [inputText, animationSpeed])

  useEffect(() => {
    if (isPlaying && currentStep < processingSteps.length) {
      const timer = setTimeout(() => {
        setProcessedSteps(prev => [...prev, processingSteps[currentStep].id])
        setCurrentStep(prev => prev + 1)
        
        if (currentStep === processingSteps.length - 1) {
          setShowPrediction(true)
          setIsPlaying(false)
        }
      }, processingSteps[currentStep]?.duration || 1000)

      return () => clearTimeout(timer)
    }
  }, [isPlaying, currentStep, processingSteps])

  const resetDemo = () => {
    setIsPlaying(false)
    setCurrentStep(0)
    setProcessedSteps([])
    setShowPrediction(false)
  }

  const togglePlayback = () => {
    if (currentStep >= processingSteps.length) {
      resetDemo()
      setTimeout(() => setIsPlaying(true), 100)
    } else {
      setIsPlaying(!isPlaying)
    }
  }

  return (
    <div className="w-full max-w-6xl mx-auto p-6 space-y-6">
      {/* Header and Input Section */}
      <Card className="border-2 border-blue-200">
        <CardHeader>
          <CardTitle className="flex items-center gap-3 text-2xl">
            <Brain className="h-8 w-8 text-blue-600" />
            How ChatGPT Processes Any Text Input
          </CardTitle>
          <CardDescription>
            Interactive demonstration of internal LLM processing from input to response generation
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Input Controls */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div>
                <Label htmlFor="input-text" className="text-sm font-medium">
                  Enter your text to analyze:
                </Label>
                <Input
                  id="input-text"
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  placeholder="Type anything you want to analyze..."
                  className="mt-1"
                />
              </div>
              
              <div className="space-y-2">
                <Label className="text-sm font-medium">Example Inputs:</Label>
                <div className="grid grid-cols-1 gap-2">
                  {EXAMPLE_INPUTS.map((example, index) => (
                    <Button
                      key={index}
                      variant="outline"
                      size="sm"
                      onClick={() => setInputText(example.text)}
                      className="justify-start text-left h-auto p-2"
                    >
                      <div>
                        <div className="font-medium text-xs">{example.category}</div>
                        <div className="text-xs text-gray-600 truncate">{example.text}</div>
                      </div>
                    </Button>
                  ))}
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <div>
                <Label className="text-sm font-medium flex items-center gap-2">
                  <Sparkles className="h-4 w-4" />
                  Animation Speed
                </Label>
                <Slider
                  value={animationSpeed}
                  onValueChange={setAnimationSpeed}
                  max={3}
                  min={0.5}
                  step={0.5}
                  className="mt-2"
                />
                <div className="text-xs text-gray-600 mt-1">
                  Current: {animationSpeed[0]}x speed
                </div>
              </div>

              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="technical-details"
                  checked={showTechnicalDetails}
                  onChange={(e) => setShowTechnicalDetails(e.target.checked)}
                  className="rounded"
                />
                <Label htmlFor="technical-details" className="text-sm">
                  Show technical details
                </Label>
              </div>
            </div>
          </div>

          {/* Real-time Analysis */}
          {analysis && (
            <Card className="bg-gradient-to-r from-blue-50 to-purple-50">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg flex items-center gap-2">
                  <Calculator className="h-5 w-5" />
                  Real-time Input Analysis
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center">
                    <Badge variant="secondary" className="mb-2">{analysis.genre}</Badge>
                    <div className="text-xs text-gray-600">Detected Genre</div>
                  </div>
                  <div className="text-center">
                    <Badge variant="outline" className="mb-2">{analysis.intent}</Badge>
                    <div className="text-xs text-gray-600">Intent</div>
                  </div>
                  <div className="text-center">
                    <Badge variant="default" className="mb-2">
                      {(analysis.confidence * 100).toFixed(1)}%
                    </Badge>
                    <div className="text-xs text-gray-600">Confidence</div>
                  </div>
                  <div className="text-center">
                    <Badge variant="destructive" className="mb-2">{analysis.sentiment}</Badge>
                    <div className="text-xs text-gray-600">Sentiment</div>
                  </div>
                </div>
                <div className="mt-4 text-sm text-gray-700">
                  <strong>Expected Response:</strong> {analysis.expectation}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Processing Controls */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-lg flex items-center gap-2">
                <Zap className="h-5 w-5" />
                Processing Simulation
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-4 mb-4">
                <Button 
                  onClick={togglePlayback}
                  variant={isPlaying ? "destructive" : "default"}
                  className="flex items-center gap-2"
                >
                  {isPlaying ? (
                    <>
                      <Pause className="h-4 w-4" />
                      Pause
                    </>
                  ) : (
                    <>
                      <Play className="h-4 w-4" />
                      {currentStep >= processingSteps.length ? 'Restart' : 'Start'} Processing
                    </>
                  )}
                </Button>
                <Button 
                  onClick={resetDemo}
                  variant="outline"
                  className="flex items-center gap-2"
                >
                  <RotateCcw className="h-4 w-4" />
                  Reset
                </Button>
                <div className="flex-1">
                  <Progress 
                    value={(processedSteps.length / processingSteps.length) * 100} 
                    className="h-2"
                  />
                </div>
                <Badge variant="outline">
                  {processedSteps.length}/{processingSteps.length}
                </Badge>
              </div>
              
              <Alert>
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>
                  {isPlaying 
                    ? `Processing step ${currentStep + 1}/${processingSteps.length}: ${processingSteps[currentStep]?.name || ''}...`
                    : processedSteps.length === 0 
                      ? "Click 'Start Processing' to begin the step-by-step analysis of your input"
                      : `Completed ${processedSteps.length}/${processingSteps.length} processing steps`
                  }
                </AlertDescription>
              </Alert>
            </CardContent>
          </Card>
        </CardContent>
      </Card>

      {/* Processing Steps Visualization */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
        {processingSteps.map((step, index) => {
          const isProcessed = processedSteps.includes(step.id)
          const isCurrent = index === currentStep && isPlaying
          
          return (
            <motion.div
              key={step.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <Card 
                className={`relative transition-all duration-500 h-full ${
                  isCurrent 
                    ? 'border-blue-500 bg-blue-50 shadow-lg ring-2 ring-blue-200' 
                    : isProcessed 
                      ? 'border-green-500 bg-green-50 shadow-md' 
                      : 'border-gray-200 hover:shadow-md'
                }`}
              >
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center gap-2">
                    {isCurrent && (
                      <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                      >
                        <Layers className="h-4 w-4 text-blue-600" />
                      </motion.div>
                    )}
                    {isProcessed && !isCurrent && (
                      <div className="h-4 w-4 bg-green-500 rounded-full flex items-center justify-center">
                        <motion.div
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                          className="h-2 w-2 bg-white rounded-full"
                        />
                      </div>
                    )}
                    {!isProcessed && !isCurrent && (
                      <div className="h-4 w-4 bg-gray-300 rounded-full" />
                    )}
                    {step.name}
                  </CardTitle>
                  <CardDescription className="text-xs">
                    {step.description}
                  </CardDescription>
                </CardHeader>
                
                <AnimatePresence>
                  {(isProcessed || isCurrent) && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={{ duration: 0.3 }}
                    >
                      <CardContent className="pt-0 space-y-3">
                        {/* Main Details */}
                        <div className="space-y-2">
                          {step.details.map((detail, detailIndex) => (
                            <motion.div
                              key={detailIndex}
                              initial={{ opacity: 0, x: -10 }}
                              animate={{ opacity: 1, x: 0 }}
                              transition={{ delay: detailIndex * 0.1 }}
                              className="text-xs text-gray-700 bg-white/60 rounded p-2 border"
                            >
                              {detail}
                            </motion.div>
                          ))}
                        </div>

                        {/* Technical Details (if enabled) */}
                        {showTechnicalDetails && step.technicalDetails && (
                          <div className="space-y-2">
                            <div className="text-xs font-semibold text-blue-700 flex items-center gap-1">
                              <Calculator className="h-3 w-3" />
                              Technical Details
                            </div>
                            {step.technicalDetails.map((detail, detailIndex) => (
                              <motion.div
                                key={detailIndex}
                                initial={{ opacity: 0, x: -10 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: detailIndex * 0.1 + 0.3 }}
                                className="text-xs text-blue-800 bg-blue-100/60 rounded p-2 border border-blue-200"
                              >
                                {detail}
                              </motion.div>
                            ))}
                          </div>
                        )}

                        {/* Examples */}
                        {step.examples && (
                          <div className="space-y-2">
                            <div className="text-xs font-semibold text-purple-700 flex items-center gap-1">
                              <Edit3 className="h-3 w-3" />
                              Examples
                            </div>
                            {step.examples.map((example, exampleIndex) => (
                              <motion.div
                                key={exampleIndex}
                                initial={{ opacity: 0, x: -10 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: exampleIndex * 0.1 + 0.5 }}
                                className="text-xs text-purple-800 bg-purple-100/60 rounded p-2 border border-purple-200"
                              >
                                {example}
                              </motion.div>
                            ))}
                          </div>
                        )}
                      </CardContent>
                    </motion.div>
                  )}
                </AnimatePresence>
              </Card>
            </motion.div>
          )
        })}
      </div>

      {/* Generated Response */}
      <AnimatePresence>
        {showPrediction && analysis && (
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -30 }}
            transition={{ duration: 0.5 }}
          >
            <Card className="border-2 border-green-200 bg-gradient-to-r from-green-50 to-emerald-50">
              <CardHeader>
                <CardTitle className="flex items-center gap-3 text-xl text-green-800">
                  <ArrowRight className="h-6 w-6" />
                  Complete Processing Result
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="bg-white rounded-lg p-4 border-2 border-green-200">
                    <div className="text-sm text-gray-600 mb-2">
                      <strong>Original Input:</strong>
                    </div>
                    <p className="text-gray-800 mb-4 font-medium">
                      "{inputText}"
                    </p>
                    <div className="text-sm text-gray-600 mb-2">
                      <strong>Generated Response:</strong>
                    </div>
                    <p className="text-green-700 font-medium leading-relaxed">
                      {generateResponse(inputText, analysis)}
                    </p>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                    <div className="bg-white/60 rounded-lg p-3 border">
                      <strong className="text-green-700">Style Consistency:</strong>
                      <div className="mt-1 text-gray-700">
                        Maintains {analysis.genre.toLowerCase()} style and tone throughout response
                      </div>
                    </div>
                    <div className="bg-white/60 rounded-lg p-3 border">
                      <strong className="text-green-700">Context Awareness:</strong>
                      <div className="mt-1 text-gray-700">
                        Responds appropriately to {analysis.intent.toLowerCase()} intent
                      </div>
                    </div>
                    <div className="bg-white/60 rounded-lg p-3 border">
                      <strong className="text-green-700">Quality Score:</strong>
                      <div className="mt-1 text-gray-700">
                        {(analysis.confidence * 100).toFixed(1)}% confidence based on pattern recognition
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
