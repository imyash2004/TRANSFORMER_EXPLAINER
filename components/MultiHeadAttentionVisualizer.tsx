'use client'

import React, { useState, useEffect, useCallback } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Slider } from '@/components/ui/slider'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Progress } from '@/components/ui/progress'
import { AlertCircle, Brain, Eye, Lightbulb, Play, Pause, RotateCcw } from 'lucide-react'
import { Alert, AlertDescription } from '@/components/ui/alert'

interface AttentionHead {
  id: number
  name: string
  specialization: string
  color: string
  weights: number[][]
  description: string
}

interface Token {
  text: string
  position: number
  embedding: number[]
  representations: { [layerIndex: number]: number[] }
}

export default function MultiHeadAttentionVisualizer() {
  const [inputText, setInputText] = useState("The cat sat on the mat and looked at the bird")
  const [numHeads, setNumHeads] = useState([8])
  const [dModel, setDModel] = useState([512])
  const [currentStep, setCurrentStep] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [selectedHead, setSelectedHead] = useState(0)
  const [showMath, setShowMath] = useState(false)
  const [animationSpeed, setAnimationSpeed] = useState([1])
  const [tokens, setTokens] = useState<Token[]>([])
  const [attentionHeads, setAttentionHeads] = useState<AttentionHead[]>([])

  // Initialize demo data
  useEffect(() => {
    const demoTokens = inputText.split(' ').map((word, i) => ({
      text: word,
      position: i,
      embedding: Array(dModel[0]).fill(0).map(() => Math.random() * 2 - 1),
      representations: {
        0: Array(dModel[0]).fill(0).map(() => Math.random() * 2 - 1)
      }
    }))
    setTokens(demoTokens)

    const heads: AttentionHead[] = [
      {
        id: 0,
        name: "Syntactic Head",
        specialization: "Subject-Verb Relations",
        color: "#3B82F6",
        weights: generateAttentionWeights(demoTokens.length, 'syntactic'),
        description: "Focuses on grammatical relationships like subject-verb, verb-object connections"
      },
      {
        id: 1,
        name: "Semantic Head",
        specialization: "Word Meaning",
        color: "#10B981",
        weights: generateAttentionWeights(demoTokens.length, 'semantic'),
        description: "Attends to semantically related words and concepts"
      },
      {
        id: 2,
        name: "Positional Head",
        specialization: "Word Order",
        color: "#F59E0B",
        weights: generateAttentionWeights(demoTokens.length, 'positional'),
        description: "Tracks position and sequence information"
      },
      {
        id: 3,
        name: "Coreference Head",
        specialization: "Pronoun Resolution",
        color: "#8B5CF6",
        weights: generateAttentionWeights(demoTokens.length, 'coreference'),
        description: "Resolves pronouns and references to their antecedents"
      },
      {
        id: 4,
        name: "Local Context Head",
        specialization: "Nearby Words",
        color: "#EF4444",
        weights: generateAttentionWeights(demoTokens.length, 'local'),
        description: "Focuses on immediate neighboring words"
      },
      {
        id: 5,
        name: "Long-Range Head",
        specialization: "Distant Dependencies",
        color: "#06B6D4",
        weights: generateAttentionWeights(demoTokens.length, 'longrange'),
        description: "Captures long-distance relationships in text"
      },
      {
        id: 6,
        name: "Copy Head",
        specialization: "Repetition Detection",
        color: "#84CC16",
        weights: generateAttentionWeights(demoTokens.length, 'copy'),
        description: "Identifies repeated words and patterns"
      },
      {
        id: 7,
        name: "Broad Head",
        specialization: "Global Context",
        color: "#F97316",
        weights: generateAttentionWeights(demoTokens.length, 'broad'),
        description: "Provides broad, distributed attention across the sequence"
      }
    ].slice(0, numHeads[0])

    setAttentionHeads(heads)
  }, [inputText, numHeads, dModel])

  // Generate attention weights based on head type
  function generateAttentionWeights(seqLen: number, type: string): number[][] {
    const weights = Array(seqLen).fill(0).map(() => Array(seqLen).fill(0))
    
    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < seqLen; j++) {
        let weight = 0
        
        switch (type) {
          case 'syntactic':
            // Higher weights for subject-verb, verb-object patterns
            if (Math.abs(i - j) === 1) weight = 0.6 + Math.random() * 0.3
            else if (i === j) weight = 0.3 + Math.random() * 0.2
            else weight = Math.random() * 0.1
            break
            
          case 'semantic':
            // Higher weights for semantically related words
            const semanticGroups = [[0, 1], [2, 3, 4], [6, 7, 8]]
            const sameGroup = semanticGroups.some(group => 
              group.includes(i) && group.includes(j)
            )
            weight = sameGroup ? 0.5 + Math.random() * 0.4 : Math.random() * 0.2
            break
            
          case 'positional':
            // Attention decreases with distance
            const distance = Math.abs(i - j)
            weight = Math.exp(-distance * 0.3) + Math.random() * 0.1
            break
            
          case 'local':
            // Strong local attention
            if (Math.abs(i - j) <= 1) weight = 0.7 + Math.random() * 0.2
            else weight = Math.random() * 0.05
            break
            
          case 'longrange':
            // Attention to distant words
            if (Math.abs(i - j) > 3) weight = 0.4 + Math.random() * 0.3
            else weight = Math.random() * 0.1
            break
            
          default:
            weight = Math.random()
        }
        
        weights[i][j] = Math.max(0, weight)
      }
      
      // Normalize to sum to 1
      const sum = weights[i].reduce((a, b) => a + b, 0)
      if (sum > 0) {
        weights[i] = weights[i].map(w => w / sum)
      }
    }
    
    return weights
  }

  // Animation control
  useEffect(() => {
    let interval: NodeJS.Timeout
    if (isPlaying) {
      interval = setInterval(() => {
        setCurrentStep(prev => (prev + 1) % 6)
      }, 2000 / animationSpeed[0])
    }
    return () => clearInterval(interval)
  }, [isPlaying, animationSpeed])

  const stepTitles = [
    "1. Input Processing & Embeddings",
    "2. Linear Projections (Q, K, V)",
    "3. Multi-Head Split & Parallel Processing",
    "4. Attention Computation & Scaling",
    "5. Head Concatenation & Integration",
    "6. Output Projection & Final Transformation"
  ]

  const AttentionMatrix = ({ head }: { head: AttentionHead }) => (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <div 
          className="w-4 h-4 rounded" 
          style={{ backgroundColor: head.color }}
        />
        <span className="font-medium">{head.name}</span>
        <Badge variant="secondary">{head.specialization}</Badge>
      </div>
      
      <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${tokens.length}, 1fr)` }}>
        {/* Header row with token names */}
        <div></div>
        {tokens.map(token => (
          <div key={`header-${token.position}`} className="text-xs text-center p-1 font-medium">
            {token.text}
          </div>
        ))}
        
        {/* Attention matrix */}
        {tokens.map((fromToken, i) => (
          <React.Fragment key={`row-${i}`}>
            <div className="text-xs p-1 font-medium">{fromToken.text}</div>
            {tokens.map((toToken, j) => (
              <div
                key={`cell-${i}-${j}`}
                className="aspect-square border border-gray-200 flex items-center justify-center text-xs rounded"
                style={{
                  backgroundColor: `${head.color}${Math.floor(head.weights[i][j] * 255).toString(16).padStart(2, '0')}`,
                  color: head.weights[i][j] > 0.5 ? 'white' : 'black'
                }}
                title={`${fromToken.text} → ${toToken.text}: ${head.weights[i][j].toFixed(3)}`}
              >
                {head.weights[i][j].toFixed(2)}
              </div>
            ))}
          </React.Fragment>
        ))}
      </div>
    </div>
  )

  const MathematicalFormulation = () => (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Mathematical Foundation
          </CardTitle>
          <CardDescription>Understanding the math behind multi-head attention</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div>
            <h4 className="font-semibold mb-3 text-lg">🧮 Complete Multi-Head Attention Formula:</h4>
            <div className="bg-gray-100 p-4 rounded font-mono text-sm space-y-2">
              <div><strong>MultiHead(Q,K,V) = Concat(head₁, head₂, ..., headₕ)W^O</strong></div>
              <div className="text-gray-600">where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)</div>
            </div>
          </div>
          
          <div>
            <h4 className="font-semibold mb-3">⚡ Core Attention Function:</h4>
            <div className="bg-blue-50 p-4 rounded space-y-2">
              <div className="font-mono text-sm">
                <strong>Attention(Q,K,V) = softmax(QK^T / √d_k)V</strong>
              </div>
              <div className="text-sm text-blue-700">
                <strong>Step by step:</strong>
                <ol className="list-decimal list-inside mt-2 space-y-1">
                  <li>Compute similarity scores: <code>QK^T</code></li>
                  <li>Scale by dimension: <code>/ √d_k</code></li>
                  <li>Apply softmax: <code>softmax(...)</code></li>
                  <li>Weight the values: <code>× V</code></li>
                </ol>
              </div>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h4 className="font-semibold mb-2">📏 Current Dimensions:</h4>
              <ul className="space-y-1 text-sm bg-green-50 p-3 rounded">
                <li>• <strong>d_model</strong> = {dModel[0]} (model dimension)</li>
                <li>• <strong>h</strong> = {numHeads[0]} (number of heads)</li>
                <li>• <strong>d_k = d_v</strong> = {Math.floor(dModel[0] / numHeads[0])} (head dimension)</li>
                <li>• <strong>sequence_length</strong> = {tokens.length}</li>
                <li>• <strong>vocab_size</strong> ≈ 50,000 (typical)</li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-semibold mb-2">⚙️ Parameter Matrices:</h4>
              <ul className="space-y-1 text-sm bg-purple-50 p-3 rounded">
                <li>• <strong>W_i^Q</strong> ∈ ℝ^{dModel[0]}×{Math.floor(dModel[0] / numHeads[0])} (Query projection)</li>
                <li>• <strong>W_i^K</strong> ∈ ℝ^{dModel[0]}×{Math.floor(dModel[0] / numHeads[0])} (Key projection)</li>
                <li>• <strong>W_i^V</strong> ∈ ℝ^{dModel[0]}×{Math.floor(dModel[0] / numHeads[0])} (Value projection)</li>
                <li>• <strong>W^O</strong> ∈ ℝ^{dModel[0]}×{dModel[0]} (Output projection)</li>
              </ul>
            </div>
          </div>

          <div>
            <h4 className="font-semibold mb-3">🎯 Why Scaling by √d_k?</h4>
            <div className="bg-yellow-50 p-4 rounded space-y-2">
              <div className="text-sm">
                <strong>Problem:</strong> Without scaling, dot products grow with dimension size
              </div>
              <div className="font-mono text-xs bg-white p-2 rounded">
                If d_k = 64, typical dot products ≈ 8 (large!)
                <br />If d_k = 512, typical dot products ≈ 23 (huge!)
              </div>
              <div className="text-sm">
                <strong>Solution:</strong> Divide by √d_k to normalize variance
              </div>
              <div className="font-mono text-xs bg-white p-2 rounded">
                After scaling: dot products ≈ 1 (manageable)
                <br />Softmax stays in its "linear" region → better gradients
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-semibold mb-3">🔢 Computational Complexity:</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-red-50 p-3 rounded">
                <div className="font-medium text-red-700">Time Complexity:</div>
                <div className="text-sm space-y-1">
                  <div>• QK^T computation: O(n² × d_k)</div>
                  <div>• Softmax: O(n²)</div>
                  <div>• Attention × V: O(n² × d_v)</div>
                  <div><strong>Total: O(n² × d_model)</strong></div>
                </div>
              </div>
              <div className="bg-blue-50 p-3 rounded">
                <div className="font-medium text-blue-700">Space Complexity:</div>
                <div className="text-sm space-y-1">
                  <div>• Attention matrices: O(h × n²)</div>
                  <div>• Intermediate results: O(n × d_model)</div>
                  <div>• Parameters: O(d_model²)</div>
                  <div><strong>Total: O(h × n² + d_model²)</strong></div>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>📚 Step-by-Step Mathematical Example</CardTitle>
          <CardDescription>Concrete numbers with our current settings</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="bg-gray-50 p-4 rounded space-y-3">
            <h5 className="font-semibold">Example Calculation for Token "cat":</h5>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              <div className="bg-white p-3 rounded border">
                <div className="font-medium text-green-600 mb-2">1. Query Vector (Q)</div>
                <div className="font-mono text-xs">
                  q_cat = [0.2, -0.1, 0.8, ...]
                  <br />Shape: (1, {Math.floor(dModel[0] / numHeads[0])})
                </div>
              </div>
              
              <div className="bg-white p-3 rounded border">
                <div className="font-medium text-blue-600 mb-2">2. Key Matrix (K)</div>
                <div className="font-mono text-xs">
                  K = [[0.1, 0.3, ...],  # the
                  <br />     [0.5, -0.2, ...], # cat  
                  <br />     [0.2, 0.7, ...]] # sat
                  <br />Shape: ({tokens.length}, {Math.floor(dModel[0] / numHeads[0])})
                </div>
              </div>
              
              <div className="bg-white p-3 rounded border">
                <div className="font-medium text-purple-600 mb-2">3. Attention Scores</div>
                <div className="font-mono text-xs">
                  scores = q_cat × K^T
                  <br />= [2.1, 0.8, 1.5, ...]
                  <br />scaled = scores / √{Math.floor(dModel[0] / numHeads[0])}
                  <br />= [0.26, 0.10, 0.19, ...]
                </div>
              </div>
            </div>
            
            <div className="bg-white p-3 rounded border">
              <div className="font-medium text-orange-600 mb-2">4. Final Attention Weights (after softmax):</div>
              <div className="font-mono text-xs">
                weights = softmax([0.26, 0.10, 0.19, ...]) = [0.32, 0.24, 0.28, 0.16]
                <br />Sum = {attentionHeads.length > 0 ? "1.00" : "1.00"} ✓ (normalized probabilities)
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )

  const ProcessingSteps = () => (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Eye className="h-5 w-5" />
          Step-by-Step Process
        </CardTitle>
        <CardDescription>
          Watch how multi-head attention processes information
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center gap-2 mb-4">
          <Button
            onClick={() => setIsPlaying(!isPlaying)}
            variant={isPlaying ? "destructive" : "default"}
            size="sm"
          >
            {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            {isPlaying ? "Pause" : "Play"}
          </Button>
          <Button
            onClick={() => setCurrentStep(0)}
            variant="outline"
            size="sm"
          >
            <RotateCcw className="h-4 w-4" />
            Reset
          </Button>
          <div className="flex-1">
            <Progress value={(currentStep + 1) * 100 / 6} className="h-2" />
          </div>
        </div>
        
        <div className="space-y-2">
          {stepTitles.map((title, index) => (
            <div
              key={index}
              className={`p-3 rounded border transition-all ${
                index === currentStep 
                  ? 'border-blue-500 bg-blue-50' 
                  : index < currentStep 
                    ? 'border-green-500 bg-green-50' 
                    : 'border-gray-200'
              }`}
            >
              <div className="flex items-center gap-2">
                <div className={`w-6 h-6 rounded-full flex items-center justify-center text-sm font-bold ${
                  index === currentStep 
                    ? 'bg-blue-500 text-white' 
                    : index < currentStep 
                      ? 'bg-green-500 text-white' 
                      : 'bg-gray-200'
                }`}>
                  {index + 1}
                </div>
                <span className="font-medium">{title}</span>
              </div>
              
              {index === currentStep && (
                <div className="mt-2 text-sm text-gray-600">
                  {getStepDescription(index)}
                </div>
              )}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )

  const getStepDescription = (step: number): string => {
    const descriptions = [
      "📝 Input tokens are converted to embeddings and positional encodings are added. This creates the foundational representation that captures both meaning and position.",
      "🔄 Each token creates Query (Q), Key (K), and Value (V) vectors through learned linear transformations. Think of Q as 'what I'm looking for', K as 'what I am', and V as 'what information I carry'.",
      "🧠 The model dimension is split across multiple attention heads for parallel processing. Each head gets a smaller dimension but can focus on different types of relationships.",
      "⚡ Each head computes attention weights using the scaled dot-product formula: softmax(QK^T/√d_k)V. The scaling prevents gradients from vanishing in high dimensions.",
      "🔗 All head outputs are concatenated back into the original model dimension. This combines all the different perspectives each head discovered.",
      "✨ A final linear transformation projects the concatenated output to create the enhanced token representation that captures multi-faceted relationships."
    ]
    return descriptions[step] || ""
  }

  const getHeadExample = (headId: number): string => {
    const examples = [
      "In 'The cat ate', this head connects 'cat' strongly with 'ate' (subject-verb relationship)",
      "In 'big enormous house', this head connects 'big' with 'enormous' (similar meanings)",
      "This head focuses on nearby words: 'the' → 'cat', 'on' → 'mat' (local relationships)", 
      "In 'John... he', this head connects 'he' back to 'John' (pronoun resolution)",
      "This head attends to immediate neighbors, creating smooth local context flow",
      "This head connects words far apart: 'cat' at position 1 with 'sleeping' at position 8",
      "In 'the cat... the cat', this head detects the repeated phrase",
      "This head distributes attention broadly, providing global sentence context"
    ]
    return examples[headId] || "Specialized attention pattern for this head type"
  }

  const analyzeAttentionPattern = (head: AttentionHead): string[] => {
    const patterns: string[] = []
    const weights = head.weights
    
    // Check for diagonal dominance (self-attention)
    const diagonalSum = weights.reduce((sum, row, i) => sum + (row[i] || 0), 0)
    if (diagonalSum / weights.length > 0.3) {
      patterns.push("Strong self-attention (tokens focus on themselves)")
    }
    
    // Check for local attention
    let localSum = 0
    for (let i = 0; i < weights.length; i++) {
      for (let j = Math.max(0, i-1); j <= Math.min(weights.length-1, i+1); j++) {
        localSum += weights[i][j] || 0
      }
    }
    if (localSum / (weights.length * 3) > 0.4) {
      patterns.push("Local attention pattern (focuses on nearby words)")
    }
    
    // Check for broadcast pattern
    const maxWeights = weights.map(row => Math.max(...row))
    const avgMax = maxWeights.reduce((a, b) => a + b, 0) / maxWeights.length
    if (avgMax > 0.6) {
      patterns.push("Focused attention (strong concentration on specific tokens)")
    } else if (avgMax < 0.3) {
      patterns.push("Distributed attention (broad, even focus)")
    }
    
    // Check for long-range dependencies
    let longRangeSum = 0
    let longRangeCount = 0
    for (let i = 0; i < weights.length; i++) {
      for (let j = 0; j < weights.length; j++) {
        if (Math.abs(i - j) > 3) {
          longRangeSum += weights[i][j] || 0
          longRangeCount++
        }
      }
    }
    if (longRangeCount > 0 && longRangeSum / longRangeCount > 0.1) {
      patterns.push("Long-range dependencies (connects distant words)")
    }
    
    return patterns.length > 0 ? patterns : ["Mixed attention pattern"]
  }

  const getStrongestConnections = (head: AttentionHead): Array<{from: string, to: string, weight: number}> => {
    const connections: Array<{from: string, to: string, weight: number}> = []
    
    for (let i = 0; i < head.weights.length && i < tokens.length; i++) {
      for (let j = 0; j < head.weights[i].length && j < tokens.length; j++) {
        if (i !== j) { // Exclude self-attention
          connections.push({
            from: tokens[i]?.text || `Token${i}`,
            to: tokens[j]?.text || `Token${j}`,
            weight: head.weights[i][j]
          })
        }
      }
    }
    
    return connections
      .sort((a, b) => b.weight - a.weight)
      .slice(0, 5) // Top 5 strongest connections
  }

  const HeadComparison = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {attentionHeads.map(head => (
        <Card 
          key={head.id} 
          className={`cursor-pointer transition-all ${
            selectedHead === head.id ? 'ring-2 ring-blue-500' : ''
          }`}
          onClick={() => setSelectedHead(head.id)}
        >
          <CardHeader className="pb-2">
            <div className="flex items-center gap-2">
              <div 
                className="w-4 h-4 rounded" 
                style={{ backgroundColor: head.color }}
              />
              <CardTitle className="text-sm">{head.name}</CardTitle>
            </div>
            <Badge variant="outline" className="text-xs">{head.specialization}</Badge>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-gray-600">{head.description}</p>
            <div className="mt-2">
              <div className="text-xs font-medium mb-1">Attention Pattern:</div>
              <div className="grid grid-cols-3 gap-1">
                {head.weights.slice(0, 3).map((row, i) => 
                  row.slice(0, 3).map((weight, j) => (
                    <div
                      key={`${i}-${j}`}
                      className="aspect-square border border-gray-200 rounded"
                      style={{
                        backgroundColor: `${head.color}${Math.floor(weight * 255).toString(16).padStart(2, '0')}`
                      }}
                    />
                  ))
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  )

  const InteractiveControls = () => (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Lightbulb className="h-5 w-5" />
          Interactive Controls
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="input-text">Input Text</Label>
          <Input
            id="input-text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Enter text to analyze..."
          />
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label>Number of Heads: {numHeads[0]}</Label>
            <Slider
              value={numHeads}
              onValueChange={setNumHeads}
              min={1}
              max={16}
              step={1}
              className="w-full"
            />
          </div>
          
          <div className="space-y-2">
            <Label>Model Dimension: {dModel[0]}</Label>
            <Slider
              value={dModel}
              onValueChange={setDModel}
              min={128}
              max={1024}
              step={64}
              className="w-full"
            />
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label>Animation Speed: {animationSpeed[0]}x</Label>
            <Slider
              value={animationSpeed}
              onValueChange={setAnimationSpeed}
              min={0.5}
              max={3}
              step={0.5}
              className="w-full"
            />
          </div>
          
          <div className="flex items-center space-x-2">
            <Switch
              id="show-math"
              checked={showMath}
              onCheckedChange={setShowMath}
            />
            <Label htmlFor="show-math">Show Mathematical Details</Label>
          </div>
        </div>
      </CardContent>
    </Card>
  )

  return (
    <div className="space-y-6">
      <div className="text-center space-y-2">
        <h1 className="text-3xl font-bold">Multi-Head Attention Mechanism</h1>
        <p className="text-lg text-gray-600">
          Interactive exploration of how transformers process information in parallel
        </p>
      </div>

      <Alert>
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>
          Multi-head attention allows the model to attend to different types of relationships 
          simultaneously. Each head specializes in different aspects of language understanding.
        </AlertDescription>
      </Alert>

      <InteractiveControls />

      <Tabs defaultValue="overview" className="w-full">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="heads">Head Analysis</TabsTrigger>
          <TabsTrigger value="process">Process Flow</TabsTrigger>
          <TabsTrigger value="math">Mathematics</TabsTrigger>
          <TabsTrigger value="comparison">Compare Heads</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>🤔 Why Multiple Heads? The Library Analogy</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="space-y-2">
                  <h4 className="font-semibold">🧠 Parallel Processing</h4>
                  <p className="text-sm text-gray-600">
                    Imagine having multiple librarians working simultaneously. Each specializes in different topics - 
                    one knows science, another history, another fiction. Multi-head attention works similarly, 
                    with each head specializing in different language relationships.
                  </p>
                </div>
                <div className="space-y-2">
                  <h4 className="font-semibold">🎯 Specialized Attention Patterns</h4>
                  <p className="text-sm text-gray-600">
                    Head 1 might focus on "Who did what?" (subject-verb), Head 2 on "What goes together?" (semantic similarity), 
                    Head 3 on "What refers to what?" (coreference). This specialization emerges naturally during training!
                  </p>
                </div>
                <div className="space-y-2">
                  <h4 className="font-semibold">📈 Richer Understanding</h4>
                  <p className="text-sm text-gray-600">
                    By combining insights from all heads, the model builds a multi-dimensional understanding. 
                    It's like getting opinions from multiple experts before making a decision.
                  </p>
                </div>
                <div className="space-y-2">
                  <h4 className="font-semibold">🔀 Computational Efficiency</h4>
                  <p className="text-sm text-gray-600">
                    Instead of one huge attention computation, we split into smaller parallel computations. 
                    This is faster and allows for more specialized learning patterns.
                  </p>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>🎭 Head Specializations in Action</CardTitle>
                <CardDescription>Real examples of what each head learns</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {attentionHeads.slice(0, 6).map(head => (
                    <div key={head.id} className="flex items-start gap-3 p-3 rounded border bg-gray-50">
                      <div 
                        className="w-4 h-4 rounded-full mt-0.5 flex-shrink-0" 
                        style={{ backgroundColor: head.color }}
                      />
                      <div className="flex-1 min-w-0">
                        <div className="font-medium text-sm">{head.name}</div>
                        <div className="text-xs text-blue-600 font-medium">{head.specialization}</div>
                        <div className="text-xs text-gray-600 mt-1">{head.description}</div>
                        <div className="text-xs text-green-600 mt-1">
                          <strong>Example:</strong> {getHeadExample(head.id)}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Real-world Examples Section */}
          <Card>
            <CardHeader>
              <CardTitle>🌍 Real-World Multi-Head Attention Examples</CardTitle>
              <CardDescription>See how different heads work together on actual sentences</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-3">
                  <h4 className="font-semibold text-green-600">Example 1: "The cat that I saw yesterday was sleeping"</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded bg-blue-500"></div>
                      <span><strong>Syntactic Head:</strong> "cat" ← → "was" (subject-verb)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded bg-purple-500"></div>
                      <span><strong>Coreference Head:</strong> "that" → "cat" (relative pronoun)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded bg-green-500"></div>
                      <span><strong>Semantic Head:</strong> "cat" ↔ "sleeping" (animal behavior)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded bg-orange-500"></div>
                      <span><strong>Temporal Head:</strong> "saw" ↔ "yesterday" (time reference)</span>
                    </div>
                  </div>
                </div>
                
                <div className="space-y-3">
                  <h4 className="font-semibold text-blue-600">Example 2: "John gave Mary the book she requested"</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded bg-blue-500"></div>
                      <span><strong>Syntactic Head:</strong> "John" → "gave" → "book" (S-V-O)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded bg-purple-500"></div>
                      <span><strong>Coreference Head:</strong> "she" → "Mary" (pronoun resolution)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded bg-cyan-500"></div>
                      <span><strong>Long-range Head:</strong> "book" ↔ "requested" (distant relation)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded bg-red-500"></div>
                      <span><strong>Local Head:</strong> "the" → "book" (determiner-noun)</span>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Interactive Learning Section */}
          <Card>
            <CardHeader>
              <CardTitle>🎓 Student Learning Tips</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-3 border rounded bg-blue-50">
                  <h4 className="font-semibold text-blue-700 mb-2">👀 Visualization Strategy</h4>
                  <p className="text-sm text-blue-600">
                    Click on different heads to see their attention patterns. Notice how each focuses on different relationships!
                  </p>
                </div>
                <div className="p-3 border rounded bg-green-50">
                  <h4 className="font-semibold text-green-700 mb-2">🔄 Experiment Mode</h4>
                  <p className="text-sm text-green-600">
                    Try different sentences. See how heads adapt to new content. Change the number of heads and observe the differences!
                  </p>
                </div>
                <div className="p-3 border rounded bg-purple-50">
                  <h4 className="font-semibold text-purple-700 mb-2">📊 Pattern Recognition</h4>
                  <p className="text-sm text-purple-600">
                    Look for patterns: diagonal matrices (local attention), scattered dots (broad attention), clusters (semantic groups).
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="heads" className="space-y-4">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-xl font-semibold">🔍 Individual Head Analysis</h3>
              <div className="text-sm text-gray-600">
                Selected: <strong>{attentionHeads[selectedHead]?.name || "No head selected"}</strong>
              </div>
            </div>
            
            {selectedHead < attentionHeads.length && (
              <>
                {/* Head Details Card */}
                <Card>
                  <CardHeader>
                    <div className="flex items-center gap-3">
                      <div 
                        className="w-6 h-6 rounded" 
                        style={{ backgroundColor: attentionHeads[selectedHead].color }}
                      />
                      <div className="flex-1">
                        <CardTitle>{attentionHeads[selectedHead].name}</CardTitle>
                        <CardDescription>{attentionHeads[selectedHead].description}</CardDescription>
                      </div>
                      <Badge variant="secondary">{attentionHeads[selectedHead].specialization}</Badge>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                      <div className="text-center p-3 border rounded">
                        <div className="text-2xl font-bold text-blue-600">
                          {(attentionHeads[selectedHead].weights.flat().reduce((a, b) => a + b, 0) / 
                            (attentionHeads[selectedHead].weights.length * attentionHeads[selectedHead].weights[0].length)).toFixed(3)}
                        </div>
                        <div className="text-sm text-gray-600">Average Attention</div>
                      </div>
                      <div className="text-center p-3 border rounded">
                        <div className="text-2xl font-bold text-green-600">
                          {Math.max(...attentionHeads[selectedHead].weights.flat()).toFixed(3)}
                        </div>
                        <div className="text-sm text-gray-600">Maximum Focus</div>
                      </div>
                      <div className="text-center p-3 border rounded">
                        <div className="text-2xl font-bold text-purple-600">
                          {attentionHeads[selectedHead].weights.length}×{attentionHeads[selectedHead].weights[0].length}
                        </div>
                        <div className="text-sm text-gray-600">Matrix Size</div>
                      </div>
                    </div>
                    
                    <div className="text-sm text-green-600 bg-green-50 p-3 rounded mb-4">
                      <strong>💡 What this head does:</strong> {getHeadExample(selectedHead)}
                    </div>
                  </CardContent>
                </Card>

                {/* Attention Matrix */}
                <Card>
                  <CardHeader>
                    <CardTitle>📊 Attention Matrix Visualization</CardTitle>
                    <CardDescription>
                      Each cell shows how much attention the row token pays to the column token. 
                      Darker colors = stronger attention.
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <AttentionMatrix head={attentionHeads[selectedHead]} />
                    
                    {/* Matrix Interpretation */}
                    <div className="mt-4 p-4 bg-gray-50 rounded">
                      <h4 className="font-semibold mb-2">🎓 How to Read This Matrix:</h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                        <div>
                          <strong>Rows:</strong> Tokens asking "What should I pay attention to?"
                          <br />
                          <strong>Columns:</strong> Tokens being attended to
                          <br />
                          <strong>Values:</strong> Attention weights (sum to 1.0 for each row)
                        </div>
                        <div>
                          <strong>Dark cells:</strong> Strong attention connections
                          <br />
                          <strong>Light cells:</strong> Weak attention connections
                          <br />
                          <strong>Diagonal:</strong> Self-attention (token attending to itself)
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Pattern Analysis */}
                <Card>
                  <CardHeader>
                    <CardTitle>🔍 Pattern Analysis</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <h4 className="font-semibold mb-2 text-blue-600">Detected Patterns:</h4>
                        <div className="space-y-2 text-sm">
                          {analyzeAttentionPattern(attentionHeads[selectedHead]).map((pattern, idx) => (
                            <div key={idx} className="flex items-center gap-2 p-2 bg-blue-50 rounded">
                              <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                              <span>{pattern}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                      
                      <div>
                        <h4 className="font-semibold mb-2 text-green-600">Strongest Connections:</h4>
                        <div className="space-y-2 text-sm">
                          {getStrongestConnections(attentionHeads[selectedHead]).map((connection, idx) => (
                            <div key={idx} className="flex items-center justify-between p-2 bg-green-50 rounded">
                              <span>{connection.from} → {connection.to}</span>
                              <Badge variant="secondary">{connection.weight.toFixed(3)}</Badge>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </>
            )}
          </div>
        </TabsContent>

        <TabsContent value="process" className="space-y-4">
          <ProcessingSteps />
        </TabsContent>

        <TabsContent value="math" className="space-y-4">
          {showMath && <MathematicalFormulation />}
          {!showMath && (
            <Alert>
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                Enable "Show Mathematical Details" in the controls to see the mathematical formulation.
              </AlertDescription>
            </Alert>
          )}
        </TabsContent>

        <TabsContent value="comparison" className="space-y-4">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-xl font-semibold">🔍 Compare All Heads</h3>
              <div className="text-sm text-gray-600">
                Click on any head to focus on its pattern. Compare how different heads see the same sentence!
              </div>
            </div>
            
            {/* Head Grid Comparison */}
            <HeadComparison />
            
            {/* Detailed Analysis */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>📊 Attention Pattern Analysis</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {attentionHeads.map(head => {
                    const avgAttention = head.weights.flat().reduce((a, b) => a + b, 0) / (head.weights.length * head.weights[0].length)
                    const maxAttention = Math.max(...head.weights.flat())
                    const entropy = -head.weights.flat().reduce((sum, weight) => 
                      weight > 0 ? sum + weight * Math.log2(weight) : sum, 0
                    ) / head.weights.length
                    
                    return (
                      <div key={head.id} className="border rounded p-3 space-y-2">
                        <div className="flex items-center gap-2">
                          <div 
                            className="w-3 h-3 rounded" 
                            style={{ backgroundColor: head.color }}
                          />
                          <span className="font-medium">{head.name}</span>
                        </div>
                        <div className="grid grid-cols-3 gap-2 text-xs">
                          <div>
                            <div className="text-gray-500">Avg Attention</div>
                            <div className="font-medium">{avgAttention.toFixed(3)}</div>
                          </div>
                          <div>
                            <div className="text-gray-500">Max Focus</div>
                            <div className="font-medium">{maxAttention.toFixed(3)}</div>
                          </div>
                          <div>
                            <div className="text-gray-500">Spread</div>
                            <div className="font-medium">{entropy > 2 ? "Broad" : entropy > 1 ? "Medium" : "Focused"}</div>
                          </div>
                        </div>
                      </div>
                    )
                  })}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>🎯 Head Specialization Summary</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="space-y-3">
                    <div className="p-3 bg-blue-50 rounded">
                      <h4 className="font-semibold text-blue-700 mb-2">Linguistic Patterns</h4>
                      <div className="text-sm space-y-1">
                        <div>• <strong>Syntactic Heads</strong>: Focus on grammar (subject-verb, etc.)</div>
                        <div>• <strong>Semantic Heads</strong>: Group related meanings together</div>
                        <div>• <strong>Coreference Heads</strong>: Link pronouns to their references</div>
                      </div>
                    </div>
                    
                    <div className="p-3 bg-green-50 rounded">
                      <h4 className="font-semibold text-green-700 mb-2">Positional Patterns</h4>
                      <div className="text-sm space-y-1">
                        <div>• <strong>Local Heads</strong>: Focus on nearby words</div>
                        <div>• <strong>Long-range Heads</strong>: Connect distant words</div>
                        <div>• <strong>Positional Heads</strong>: Track word order and sequence</div>
                      </div>
                    </div>
                    
                    <div className="p-3 bg-purple-50 rounded">
                      <h4 className="font-semibold text-purple-700 mb-2">Special Functions</h4>
                      <div className="text-sm space-y-1">
                        <div>• <strong>Copy Heads</strong>: Detect repetitions and patterns</div>
                        <div>• <strong>Broad Heads</strong>: Provide global context</div>
                        <div>• <strong>Task-specific Heads</strong>: Adapt to specific tasks</div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Interactive Insights */}
            <Card>
              <CardHeader>
                <CardTitle>💡 Key Insights for Students</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  <div className="p-4 border rounded bg-gradient-to-br from-blue-50 to-blue-100">
                    <h4 className="font-semibold text-blue-700 mb-2">🎭 Emergent Specialization</h4>
                    <p className="text-sm text-blue-600">
                      Heads aren't programmed to specialize - they learn these patterns naturally during training! 
                      Each discovers its own "expertise" automatically.
                    </p>
                  </div>
                  
                  <div className="p-4 border rounded bg-gradient-to-br from-green-50 to-green-100">
                    <h4 className="font-semibold text-green-700 mb-2">🔄 Complementary Learning</h4>
                    <p className="text-sm text-green-600">
                      Heads work together, not in isolation. One head's strength covers another's weakness, 
                      creating a robust understanding system.
                    </p>
                  </div>
                  
                  <div className="p-4 border rounded bg-gradient-to-br from-purple-50 to-purple-100">
                    <h4 className="font-semibold text-purple-700 mb-2">📈 Scalable Intelligence</h4>
                    <p className="text-sm text-purple-600">
                      More heads = more perspectives = richer understanding. 
                      But there's a sweet spot - too many heads can be redundant!
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
