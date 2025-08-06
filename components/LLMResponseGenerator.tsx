"use client"

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card"
import { Button } from "./ui/button"
import { Badge } from "./ui/badge"
import { Progress } from "./ui/progress"
import { Alert, AlertDescription } from "./ui/alert"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs"
import { AlertCircle, Brain, Calculator, Layers, Zap, Play, Pause, RotateCcw, ArrowRight } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"

interface ProcessingStep {
  id: string
  name: string
  description: string
  status: 'pending' | 'processing' | 'complete'
  duration: number
  details: string[]
}

const STORY_EXAMPLES = [
  {
    input: "Once upon a time in a magical forest",
    analysis: {
      genre: "Fantasy Story",
      intent: "Story Beginning",
      expectation: "Narrative continuation with characters and plot",
      confidence: 0.95
    },
    processing_steps: [
      {
        id: "tokenization",
        name: "1. Tokenization",
        description: "Breaking text into tokens",
        status: 'pending' as const,
        duration: 500,
        details: [
          "Input: 'Once upon a time in a magical forest'",
          "Tokens: ['Once', 'upon', 'a', 'time', 'in', 'a', 'magical', 'forest']",
          "Token count: 8 tokens",
          "Special tokens: [BOS] added at beginning"
        ]
      },
      {
        id: "embeddings",
        name: "2. Token Embeddings", 
        description: "Converting tokens to numerical vectors",
        status: 'pending' as const,
        duration: 800,
        details: [
          "'Once' → [0.2, 0.8, 0.1, ...] (4096 dimensions)",
          "'upon' → [0.25, 0.85, 0.15, ...]", 
          "'magical' → [0.7, 0.3, 0.9, ...] (high fantasy score)",
          "Each token becomes a 4096-dimensional vector"
        ]
      },
      {
        id: "attention",
        name: "3. Multi-Head Attention",
        description: "Understanding relationships between tokens",
        status: 'pending' as const,
        duration: 1200,
        details: [
          "Head 1: 'Once' + 'upon' + 'time' → Story pattern (98% confidence)",
          "Head 2: 'magical' + 'forest' → Fantasy setting (94% confidence)",
          "Head 3: Positional relationships and grammar",
          "32 attention heads working in parallel"
        ]
      },
      {
        id: "context",
        name: "4. Context Understanding",
        description: "Building semantic understanding",
        status: 'pending' as const,
        duration: 1000,
        details: [
          "Genre detected: Fantasy/Fairy tale",
          "Narrative mode: Third-person storytelling",
          "Setting: Magical/supernatural environment", 
          "Expected elements: Characters, conflict, adventure"
        ]
      },
      {
        id: "prediction",
        name: "5. Next Token Prediction",
        description: "Generating probability distribution for next words",
        status: 'pending' as const,
        duration: 900,
        details: [
          "Top predictions:",
          "• 'there' (23.4%) - introduces character/element",
          "• 'lived' (18.7%) - classic story continuation",
          "• 'where' (15.2%) - setting description",
          "• 'a' (12.8%) - article before character"
        ]
      },
      {
        id: "selection",
        name: "6. Token Selection",
        description: "Choosing the next word using sampling",
        status: 'pending' as const,
        duration: 300,
        details: [
          "Using temperature sampling (T=0.7)",
          "Selected: 'there' (23.4% probability)",
          "Alternative sampling methods:",
          "• Greedy: Always highest probability",
          "• Top-k: From top 40 candidates"
        ]
      },
      {
        id: "output",
        name: "7. Response Generation",
        description: "Continuing the generation process",
        status: 'pending' as const,
        duration: 2000,
        details: [
          "Generated continuation:",
          "'there lived a wise old owl named Luna...'",
          "Process repeats for each new token",
          "Stop when [EOS] token or max length reached"
        ]
      }
    ],
    predicted_response: "there lived a wise old owl named Luna, who possessed ancient magic and watched over all the creatures of the enchanted woodland. One day, a young adventurer stumbled into her domain, seeking help to save their village from a terrible curse..."
  }
]

export default function LLMResponseGenerator() {
  const [currentExample] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentStep, setCurrentStep] = useState(0)
  const [processedSteps, setProcessedSteps] = useState<string[]>([])
  const [showPrediction, setShowPrediction] = useState(false)

  const example = STORY_EXAMPLES[currentExample]

  useEffect(() => {
    if (isPlaying && currentStep < example.processing_steps.length) {
      const timer = setTimeout(() => {
        setProcessedSteps(prev => [...prev, example.processing_steps[currentStep].id])
        setCurrentStep(prev => prev + 1)
        
        if (currentStep === example.processing_steps.length - 1) {
          setShowPrediction(true)
          setIsPlaying(false)
        }
      }, example.processing_steps[currentStep].duration)

      return () => clearTimeout(timer)
    }
  }, [isPlaying, currentStep, example.processing_steps])

  const resetDemo = () => {
    setIsPlaying(false)
    setCurrentStep(0)
    setProcessedSteps([])
    setShowPrediction(false)
  }

  const togglePlayback = () => {
    if (currentStep >= example.processing_steps.length) {
      resetDemo()
    } else {
      setIsPlaying(!isPlaying)
    }
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span className="flex items-center space-x-2">
              <Brain className="h-6 w-6" />
              <span>How ChatGPT Processes: "{example.input}"</span>
            </span>
            <div className="flex items-center space-x-2">
              <Button onClick={togglePlayback} size="sm">
                {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                {isPlaying ? 'Pause' : currentStep === 0 ? 'Start Demo' : 'Resume'}
              </Button>
              <Button onClick={resetDemo} variant="outline" size="sm">
                <RotateCcw className="h-4 w-4" />
                Reset
              </Button>
            </div>
          </CardTitle>
          <CardDescription>
            Watch the complete internal process from input text to generated response
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* Input Analysis */}
            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <h4 className="font-medium mb-2">Initial Analysis</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <span className="text-gray-600">Genre:</span>
                  <div className="font-medium">{example.analysis.genre}</div>
                </div>
                <div>
                  <span className="text-gray-600">Intent:</span>
                  <div className="font-medium">{example.analysis.intent}</div>
                </div>
                <div>
                  <span className="text-gray-600">Confidence:</span>
                  <div className="font-medium">{(example.analysis.confidence * 100).toFixed(1)}%</div>
                </div>
                <div>
                  <span className="text-gray-600">Expected:</span>
                  <div className="font-medium">{example.analysis.expectation}</div>
                </div>
              </div>
            </div>

            {/* Progress Bar */}
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Processing Progress</span>
                <span>{currentStep}/{example.processing_steps.length} steps</span>
              </div>
              <Progress value={(currentStep / example.processing_steps.length) * 100} className="h-2" />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Processing Steps */}
      <div className="grid gap-4">
        {example.processing_steps.map((step, index) => {
          const isActive = index === currentStep && isPlaying
          const isCompleted = processedSteps.includes(step.id)
          const isPending = index > currentStep

          return (
            <motion.div
              key={step.id}
              initial={{ opacity: 0.5, x: -20 }}
              animate={{ 
                opacity: isCompleted ? 1 : isPending ? 0.5 : 1,
                x: 0,
                scale: isActive ? 1.02 : 1
              }}
              transition={{ duration: 0.3 }}
            >
              <Card className={`relative ${isActive ? 'ring-2 ring-blue-500 shadow-lg' : isCompleted ? 'bg-green-50 dark:bg-green-900/20' : ''}`}>
                <CardHeader className="pb-3">
                  <CardTitle className="flex items-center justify-between text-lg">
                    <span className="flex items-center space-x-3">
                      <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-medium ${
                        isCompleted ? 'bg-green-500' : isActive ? 'bg-blue-500' : 'bg-gray-400'
                      }`}>
                        {index + 1}
                      </div>
                      <span>{step.name}</span>
                    </span>
                    <div className="flex items-center space-x-2">
                      <Badge variant={isCompleted ? "default" : isPending ? "secondary" : "outline"}>
                        {isCompleted ? 'Complete' : isActive ? 'Processing...' : 'Pending'}
                      </Badge>
                      {isActive && <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />}
                    </div>
                  </CardTitle>
                  <CardDescription>{step.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {step.details.map((detail, detailIndex) => (
                      <motion.div
                        key={detailIndex}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: isCompleted || isActive ? 1 : 0.5 }}
                        transition={{ delay: detailIndex * 0.1 }}
                        className={`text-sm p-2 rounded ${detail.includes('→') ? 'bg-gray-100 dark:bg-gray-800 font-mono' : ''}`}
                      >
                        {detail.includes('•') ? (
                          <div className="flex items-start space-x-2">
                            <span className="text-blue-500 mt-1">•</span>
                            <span>{detail.replace('•', '').trim()}</span>
                          </div>
                        ) : (
                          detail
                        )}
                      </motion.div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          )
        })}
      </div>

      {/* Generated Response */}
      <AnimatePresence>
        {showPrediction && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <Card className="border-green-200 bg-green-50 dark:bg-green-900/20">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2 text-green-800 dark:text-green-300">
                  <ArrowRight className="h-5 w-5" />
                  <span>Generated Response</span>
                </CardTitle>
                <CardDescription>
                  This is what ChatGPT would generate after processing your input
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="p-4 bg-white dark:bg-gray-800 rounded-lg border-l-4 border-green-500">
                    <div className="text-sm text-gray-600 mb-2">Original Input:</div>
                    <div className="font-medium mb-4">"{example.input}"</div>
                    
                    <div className="text-sm text-gray-600 mb-2">ChatGPT's Response:</div>
                    <div className="text-lg leading-relaxed">
                      <span className="font-medium text-blue-600">"{example.input}</span> {example.predicted_response}"
                    </div>
                  </div>

                  <Alert>
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>
                      <strong>How this works:</strong> The model continues from your input by predicting one token at a time, 
                      using the context of all previous tokens to maintain coherence and style. Each new token influences 
                      the probability distribution for the next token, creating a flowing narrative.
                    </AlertDescription>
                  </Alert>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
