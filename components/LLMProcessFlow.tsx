"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import { motion, AnimatePresence } from "framer-motion"
import { 
  Play, 
  Pause, 
  RotateCcw, 
  ArrowRight, 
  Layers, 
  Brain, 
  Zap, 
  Target,
  Calculator,
  Info
} from "lucide-react"

interface ProcessStep {
  id: number
  title: string
  description: string
  details: string[]
  icon: React.ReactNode
  color: string
  mathematics?: string
  example?: string
}

const processSteps: ProcessStep[] = [
  {
    id: 1,
    title: "Input Processing",
    description: "Text is converted to numerical tokens",
    details: [
      'User input: "The weather is"',
      'Tokenization: ["The", " weather", " is"]',
      'Token IDs: [464, 6193, 318]',
      'Vocabulary lookup from trained BPE tokenizer'
    ],
    icon: <Target className="h-5 w-5" />,
    color: "bg-blue-500",
    mathematics: 'token_ids = tokenizer.encode(text)',
    example: '"The weather is" → [464, 6193, 318]'
  },
  {
    id: 2,
    title: "Embedding Lookup",
    description: "Tokens become high-dimensional vectors",
    details: [
      'Each token ID maps to learned vector',
      'Embedding matrix: [vocab_size × embedding_dim]',
      'GPT-4: ~100K tokens × 12,288 dimensions',
      'Vectors encode semantic meaning and relationships'
    ],
    icon: <Calculator className="h-5 w-5" />,
    color: "bg-purple-500",
    mathematics: 'embeddings = E[token_ids] # E shape: [100K, 12288]',
    example: 'Token 464 → [0.1, -0.3, 0.7, ..., 0.2] (12,288 numbers)'
  },
  {
    id: 3,
    title: "Positional Encoding",
    description: "Add position information to embeddings",
    details: [
      'Pure embeddings lack word order information',
      'Sinusoidal or learned positional encodings',
      'Added to token embeddings element-wise',
      'Enables understanding of sequence structure'
    ],
    icon: <Layers className="h-5 w-5" />,
    color: "bg-green-500",
    mathematics: 'pos_emb = positional_encoding(position, d_model)',
    example: 'Position 1: [0.0, 0.1, 0.0, ...] added to embedding'
  },
  {
    id: 4,
    title: "Multi-Head Attention",
    description: "Tokens attend to each other via QKV mechanism",
    details: [
      'Each token creates Query, Key, Value vectors',
      'Attention weights = softmax(QK^T/√d_k)',
      'Multiple heads focus on different relationships',
      'Parallel processing of all token pairs'
    ],
    icon: <Brain className="h-5 w-5" />,
    color: "bg-red-500",
    mathematics: 'Attention(Q,K,V) = softmax(QK^T/√d_k)V',
    example: '"weather" attends 70% to "is", 20% to "The", 10% to others'
  },
  {
    id: 5,
    title: "Feed-Forward Network",
    description: "Non-linear transformation of representations",
    details: [
      'Two linear layers with ReLU activation',
      'Expands dimension then contracts back',
      'Processes each position independently',
      'Adds non-linearity and pattern recognition'
    ],
    icon: <Zap className="h-5 w-5" />,
    color: "bg-yellow-500",
    mathematics: 'FFN(x) = max(0, xW₁ + b₁)W₂ + b₂',
    example: '[768] → [3072] → ReLU → [768] dimensions'
  },
  {
    id: 6,
    title: "Layer Repetition",
    description: "Process repeats through multiple transformer layers",
    details: [
      'Each layer refines token representations',
      'Early layers: syntax and local patterns',
      'Middle layers: semantic relationships',
      'Late layers: high-level reasoning and abstractions'
    ],
    icon: <Layers className="h-5 w-5" />,
    color: "bg-indigo-500",
    mathematics: 'layer_i = LayerNorm(x + MultiHeadAttention(x)) + FFN(x)',
    example: 'GPT-4 has 96 layers, GPT-3 has 96, BERT has 24'
  },
  {
    id: 7,
    title: "Final Prediction",
    description: "Convert final representations to token probabilities",
    details: [
      'Linear projection to vocabulary size',
      'Softmax to convert to probabilities',
      'Select most likely next token',
      'Can use sampling for creativity'
    ],
    icon: <Target className="h-5 w-5" />,
    color: "bg-orange-500",
    mathematics: 'logits = final_hidden_state @ W_out; probs = softmax(logits)',
    example: '"sunny": 23%, "cloudy": 19%, "nice": 15%, ...'
  }
]

interface LLMProcessFlowProps {
  currentText?: string
}

export function LLMProcessFlow({ currentText = "The weather is" }: LLMProcessFlowProps) {
  const [currentStep, setCurrentStep] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [selectedStep, setSelectedStep] = useState<number | null>(null)

  // Auto-advance steps when playing
  useEffect(() => {
    if (isPlaying) {
      const interval = setInterval(() => {
        setCurrentStep(prev => {
          if (prev >= processSteps.length - 1) {
            setIsPlaying(false)
            return prev
          }
          return prev + 1
        })
      }, 2000)
      return () => clearInterval(interval)
    }
  }, [isPlaying])

  const handlePlayToggle = () => {
    setIsPlaying(!isPlaying)
  }

  const handleReset = () => {
    setCurrentStep(0)
    setIsPlaying(false)
    setSelectedStep(null)
  }

  const handleStepClick = (stepIndex: number) => {
    setCurrentStep(stepIndex)
    setSelectedStep(selectedStep === stepIndex ? null : stepIndex)
    setIsPlaying(false)
  }

  return (
    <div className="space-y-6">
      {/* Header Controls */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center space-x-2">
              <Brain className="h-6 w-6" />
              <span>LLM Processing Pipeline: From Text to Prediction</span>
            </CardTitle>
            <div className="flex items-center space-x-2">
              <Button variant="outline" size="sm" onClick={handlePlayToggle}>
                {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                {isPlaying ? "Pause" : "Play"}
              </Button>
              <Button variant="outline" size="sm" onClick={handleReset}>
                <RotateCcw className="h-4 w-4" />
                Reset
              </Button>
              <Badge variant="outline">
                Step {currentStep + 1} of {processSteps.length}
              </Badge>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <div className="flex items-center space-x-2 mb-2">
                <Info className="h-4 w-4 text-blue-600" />
                <span className="font-medium">Current Input Text:</span>
              </div>
              <div className="font-mono text-lg bg-white dark:bg-gray-700 p-3 rounded border">
                "{currentText}"
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-300 mt-2">
                Follow this text through each stage of transformer processing
              </div>
            </div>

            <Progress value={((currentStep + 1) / processSteps.length) * 100} className="w-full" />
          </div>
        </CardContent>
      </Card>

      {/* Process Flow Visualization */}
      <div className="grid grid-cols-1 lg:grid-cols-7 gap-4">
        {processSteps.map((step, index) => (
          <motion.div
            key={step.id}
            className={`relative cursor-pointer transition-all duration-300 ${
              index <= currentStep ? 'opacity-100' : 'opacity-40'
            } ${selectedStep === index ? 'scale-105' : 'hover:scale-102'}`}
            onClick={() => handleStepClick(index)}
            initial={{ y: 20, opacity: 0 }}
            animate={{ 
              y: 0, 
              opacity: index <= currentStep ? 1 : 0.4,
              scale: selectedStep === index ? 1.05 : 1
            }}
            transition={{ delay: index * 0.1 }}
          >
            <Card className={`h-full ${selectedStep === index ? 'ring-2 ring-blue-500' : ''}`}>
              <CardHeader className="pb-2">
                <div className="flex items-center space-x-2">
                  <div className={`p-2 rounded-full text-white ${step.color}`}>
                    {step.icon}
                  </div>
                  <div className="flex-1">
                    <div className="font-medium text-sm">{step.title}</div>
                    <Badge variant="outline" className="text-xs">
                      {index + 1}
                    </Badge>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="pt-0">
                <p className="text-xs text-gray-600 dark:text-gray-300 mb-2">
                  {step.description}
                </p>
                {index <= currentStep && (
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    className="absolute -top-2 -right-2 bg-green-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-xs"
                  >
                    ✓
                  </motion.div>
                )}
              </CardContent>
            </Card>

            {/* Arrow connector */}
            {index < processSteps.length - 1 && (
              <div className="hidden lg:block absolute -right-6 top-1/2 transform -translate-y-1/2">
                <ArrowRight 
                  className={`h-5 w-5 ${
                    index < currentStep ? 'text-green-500' : 'text-gray-300'
                  }`} 
                />
              </div>
            )}
          </motion.div>
        ))}
      </div>

      {/* Detailed Step Information */}
      <AnimatePresence>
        {selectedStep !== null && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-3">
                  <div className={`p-2 rounded-full text-white ${processSteps[selectedStep].color}`}>
                    {processSteps[selectedStep].icon}
                  </div>
                  <span>Step {selectedStep + 1}: {processSteps[selectedStep].title}</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Tabs defaultValue="details" className="w-full">
                  <TabsList className="grid w-full grid-cols-3">
                    <TabsTrigger value="details">Details</TabsTrigger>
                    <TabsTrigger value="mathematics">Mathematics</TabsTrigger>
                    <TabsTrigger value="example">Example</TabsTrigger>
                  </TabsList>

                  <TabsContent value="details" className="space-y-3">
                    <div className="space-y-2">
                      {processSteps[selectedStep].details.map((detail, index) => (
                        <div key={index} className="flex items-start space-x-2">
                          <div className="w-2 h-2 rounded-full bg-blue-500 mt-2 flex-shrink-0" />
                          <span className="text-sm">{detail}</span>
                        </div>
                      ))}
                    </div>
                  </TabsContent>

                  <TabsContent value="mathematics" className="space-y-3">
                    <div className="space-y-3">
                      <div>
                        <h4 className="font-medium mb-2">Mathematical Formula:</h4>
                        <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded font-mono text-sm">
                          {processSteps[selectedStep].mathematics}
                        </div>
                      </div>
                      
                      <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                        <h5 className="font-medium mb-2">Explanation:</h5>
                        <div className="text-sm space-y-1">
                          {selectedStep === 0 && (
                            <div>Tokenizer converts text strings to integer IDs using learned vocabulary</div>
                          )}
                          {selectedStep === 1 && (
                            <div>Embedding matrix E maps each token ID to a dense vector representation</div>
                          )}
                          {selectedStep === 2 && (
                            <div>Positional encoding adds location information to maintain sequence order</div>
                          )}
                          {selectedStep === 3 && (
                            <div>Self-attention computes weighted combinations based on query-key similarities</div>
                          )}
                          {selectedStep === 4 && (
                            <div>Feed-forward network applies non-linear transformations to each position</div>
                          )}
                          {selectedStep === 5 && (
                            <div>Residual connections and layer normalization stabilize deep network training</div>
                          )}
                          {selectedStep === 6 && (
                            <div>Final linear layer projects hidden states to vocabulary logits for prediction</div>
                          )}
                        </div>
                      </div>
                    </div>
                  </TabsContent>

                  <TabsContent value="example" className="space-y-3">
                    <div className="space-y-3">
                      <div>
                        <h4 className="font-medium mb-2">Concrete Example with "{currentText}":</h4>
                        <div className="bg-green-50 dark:bg-green-900/20 p-3 rounded">
                          <div className="font-mono text-sm">
                            {processSteps[selectedStep].example}
                          </div>
                        </div>
                      </div>

                      <div className="p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded">
                        <h5 className="font-medium mb-2">Real-World Scale:</h5>
                        <div className="text-sm space-y-1">
                          {selectedStep === 0 && (
                            <div>GPT-4 tokenizer has ~100,000 possible tokens in its vocabulary</div>
                          )}
                          {selectedStep === 1 && (
                            <div>GPT-4 embeddings: 100,000 × 12,288 = 1.2 billion parameters just for embeddings!</div>
                          )}
                          {selectedStep === 2 && (
                            <div>Positional encodings extend up to context lengths of 32,000+ tokens</div>
                          )}
                          {selectedStep === 3 && (
                            <div>GPT-4 has 128 attention heads across 96 layers = 12,288 total attention computations</div>
                          )}
                          {selectedStep === 4 && (
                            <div>Feed-forward networks typically expand to 4× the embedding dimension (49,152 for GPT-4)</div>
                          )}
                          {selectedStep === 5 && (
                            <div>96 transformer layers each refining representations through this process</div>
                          )}
                          {selectedStep === 6 && (
                            <div>Final prediction considers all 100,000 vocabulary tokens simultaneously</div>
                          )}
                        </div>
                      </div>
                    </div>
                  </TabsContent>
                </Tabs>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Summary Card */}
      <Card>
        <CardHeader>
          <CardTitle>Key Insights About LLM Processing</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-3">
              <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                <h4 className="font-medium text-blue-800 dark:text-blue-200">Parallel Processing</h4>
                <p className="text-sm text-blue-700 dark:text-blue-300">
                  Unlike humans reading word-by-word, LLMs process entire sequences simultaneously through matrix operations.
                </p>
              </div>
              <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                <h4 className="font-medium text-green-800 dark:text-green-200">Attention Mechanism</h4>
                <p className="text-sm text-green-700 dark:text-green-300">
                  Each token can attend to any other token, enabling understanding of long-range dependencies and complex relationships.
                </p>
              </div>
            </div>
            <div className="space-y-3">
              <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                <h4 className="font-medium text-purple-800 dark:text-purple-200">Learned Representations</h4>
                <p className="text-sm text-purple-700 dark:text-purple-300">
                  Every number in embeddings and weights is learned from training data, encoding patterns discovered in text.
                </p>
              </div>
              <div className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded">
                <h4 className="font-medium text-orange-800 dark:text-orange-200">Emergent Understanding</h4>
                <p className="text-sm text-orange-700 dark:text-orange-300">
                  Complex reasoning emerges from the interaction of simple mathematical operations across many layers.
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
