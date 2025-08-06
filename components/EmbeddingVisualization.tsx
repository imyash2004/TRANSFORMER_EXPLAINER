"use client"

import React, { useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { Sphere, Text, OrbitControls, Line } from '@react-three/drei'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card"
import { Button } from "./ui/button"
import { Badge } from "./ui/badge"
import { Progress } from "./ui/progress"
import { Alert, AlertDescription } from "./ui/alert"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs"
import { AlertCircle, Brain, Calculator, Layers, Zap, Target } from "lucide-react"
import type { TokenData } from "@/lib/textAnalyzer"
import { motion } from "framer-motion"

interface EmbeddingVisualizationProps {
  tokens: TokenData[]
  selectedToken: number | null
  onTokenSelect: (tokenId: number | null) => void
}

// Famous word embedding relationships for demonstration
// Semantic relationships for storytelling pattern visualization
const SEMANTIC_RELATIONSHIPS = [
  // Story opening pattern relationships
  { token1: "once", token2: "upon", type: "story_pattern", color: "#4CAF50" },
  { token1: "upon", token2: "time", type: "story_pattern", color: "#4CAF50" },
  { token1: "once", token2: "time", type: "story_pattern", color: "#8BC34A" },
  
  // Setting establishment relationships  
  { token1: "magical", token2: "forest", type: "setting", color: "#9C27B0" },
  { token1: "time", token2: "magical", type: "transition", color: "#FF9800" },
  { token1: "forest", token2: "[PREDICT]", type: "continuation", color: "#F44336" },
  
  // Narrative flow relationships
  { token1: "in", token2: "magical", type: "location", color: "#607D8B" },
  { token1: "magical", token2: "[PREDICT]", type: "expectation", color: "#E91E63" },
  
  // Low-importance word clustering
  { token1: "a", token2: "a", type: "function_words", color: "#9E9E9E" },
];

// Demo tokens for semantic relationship visualization - representing "once upon a time" processing
const DEMO_TOKENS = [
  { id: 10, text: "once", embedding: [0.2, 0.8, 0.1], similarity: 0.9, color: "#4CAF50", pos_tag: "ADV", semantic_role: "temporal_marker", frequency: 30, context_similarity: 0.9, explanation: "Story beginning marker - signals narrative mode", sentiment: 0.1 },
  { id: 11, text: "upon", embedding: [0.25, 0.85, 0.15], similarity: 0.95, color: "#8BC34A", pos_tag: "PREP", semantic_role: "story_connector", frequency: 20, context_similarity: 0.95, explanation: "Connects 'once' to 'time' in storytelling", sentiment: 0.1 },
  { id: 12, text: "a", embedding: [0.1, 0.1, 0.1], similarity: 0.3, color: "#9E9E9E", pos_tag: "DET", semantic_role: "determiner", frequency: 1000, context_similarity: 0.3, explanation: "Article - low semantic value in context", sentiment: 0.0 },
  { id: 13, text: "time", embedding: [0.3, 0.9, 0.2], similarity: 0.92, color: "#FF9800", pos_tag: "NOUN", semantic_role: "temporal_concept", frequency: 45, context_similarity: 0.92, explanation: "Completes classic story opening phrase", sentiment: 0.0 },
  { id: 14, text: "in", embedding: [0.15, 0.2, 0.8], similarity: 0.4, color: "#607D8B", pos_tag: "PREP", semantic_role: "location_marker", frequency: 200, context_similarity: 0.4, explanation: "Prepares for setting description", sentiment: 0.0 },
  { id: 15, text: "a", embedding: [0.1, 0.1, 0.1], similarity: 0.3, color: "#9E9E9E", pos_tag: "DET", semantic_role: "determiner", frequency: 1000, context_similarity: 0.3, explanation: "Second article - also low semantic value", sentiment: 0.0 },
  { id: 16, text: "magical", embedding: [0.7, 0.3, 0.9], similarity: 0.88, color: "#9C27B0", pos_tag: "ADJ", semantic_role: "setting_descriptor", frequency: 8, context_similarity: 0.88, explanation: "Fantasy genre indicator - triggers imaginative context", sentiment: 0.8 },
  { id: 17, text: "forest", embedding: [0.6, 0.7, 0.4], similarity: 0.85, color: "#4CAF50", pos_tag: "NOUN", semantic_role: "setting_location", frequency: 15, context_similarity: 0.85, explanation: "Natural setting - combines with 'magical' for fantasy world", sentiment: 0.3 },
  { id: 18, text: "[PREDICT]", embedding: [0.5, 0.5, 0.5], similarity: 0.0, color: "#F44336", pos_tag: "PRED", semantic_role: "next_token", frequency: 0, context_similarity: 0.0, explanation: "Model prediction point - what comes next?", sentiment: 0.0 },
];

export default function EmbeddingVisualization() {
  const [selectedToken, setSelectedToken] = useState<number | null>(null);
  const [demoMode, setDemoMode] = useState(false);
  const [selectedRelationship, setSelectedRelationship] = useState<string>("all");
  const [showConnections, setShowConnections] = useState(false);
  const [dimension, setDimension] = useState<"2d" | "3d">("3d");

  // Sample tokens for the visualization
  const tokens = [
    { id: 1, text: "the", embedding: [0.1, 0.2, 0.3], similarity: 0.9, color: "#4A90E2", pos_tag: "DET", semantic_role: "determiner", frequency: 100, context_similarity: 0.8, explanation: "Most common English determiner", sentiment: 0.0 },
    { id: 2, text: "cat", embedding: [0.7, 0.8, 0.2], similarity: 0.8, color: "#F5A623", pos_tag: "NOUN", semantic_role: "subject", frequency: 25, context_similarity: 0.7, explanation: "Domestic feline animal", sentiment: 0.7 },
    { id: 3, text: "sat", embedding: [0.3, 0.5, 0.7], similarity: 0.7, color: "#50E3C2", pos_tag: "VERB", semantic_role: "predicate", frequency: 15, context_similarity: 0.6, explanation: "Past tense of sit", sentiment: 0.0 },
    { id: 4, text: "on", embedding: [0.2, 0.1, 0.8], similarity: 0.6, color: "#BD10E0", pos_tag: "PREP", semantic_role: "preposition", frequency: 80, context_similarity: 0.5, explanation: "Preposition indicating position", sentiment: 0.0 },
    { id: 5, text: "mat", embedding: [0.6, 0.3, 0.4], similarity: 0.5, color: "#D0021B", pos_tag: "NOUN", semantic_role: "object", frequency: 5, context_similarity: 0.4, explanation: "Flat piece of material", sentiment: 0.0 },
    { id: 6, text: "king", embedding: [0.8, 0.1, 0.9], similarity: 0.9, color: "#B8860B", pos_tag: "NOUN", semantic_role: "subject", frequency: 12, context_similarity: 0.9, explanation: "Male monarch", sentiment: 0.6 },
    { id: 7, text: "queen", embedding: [0.9, 0.2, 0.8], similarity: 0.9, color: "#DA70D6", pos_tag: "NOUN", semantic_role: "subject", frequency: 10, context_similarity: 0.9, explanation: "Female monarch", sentiment: 0.6 },
    { id: 8, text: "man", embedding: [0.4, 0.6, 0.1], similarity: 0.8, color: "#4682B4", pos_tag: "NOUN", semantic_role: "subject", frequency: 30, context_similarity: 0.8, explanation: "Adult human male", sentiment: 0.0 },
    { id: 9, text: "woman", embedding: [0.5, 0.7, 0.2], similarity: 0.8, color: "#FF69B4", pos_tag: "NOUN", semantic_role: "subject", frequency: 28, context_similarity: 0.8, explanation: "Adult human female", sentiment: 0.0 },
  ];

  // Token selection handler  
  const onTokenSelect = (tokenId: number | null) => {
    setSelectedToken(tokenId);
  };

  // Use demo tokens when in demo mode, otherwise use provided tokens
  const displayTokens = demoMode ? DEMO_TOKENS : tokens

  // Calculate similarity between tokens
  const calculateSimilarity = (embedding1: number[], embedding2: number[]): number => {
    const dotProduct = embedding1.reduce((sum, val, i) => sum + val * embedding2[i], 0)
    const magnitude1 = Math.sqrt(embedding1.reduce((sum, val) => sum + val * val, 0))
    const magnitude2 = Math.sqrt(embedding2.reduce((sum, val) => sum + val * val, 0))
    return dotProduct / (magnitude1 * magnitude2)
  }

  // Enhanced 3D visualization component with relationship lines
  function Enhanced3DTokenCloud() {
    const filteredTokens = selectedRelationship === "all" 
      ? displayTokens 
      : displayTokens.filter((token: any) => {
          const relationships = SEMANTIC_RELATIONSHIPS.filter((rel: any) => 
            rel.type === selectedRelationship && 
            (rel.token1 === token.text || rel.token2 === token.text)
          );
          return relationships.length > 0;
        });

    return (
      <Canvas camera={{ position: [0, 0, 8] }}>
        <ambientLight intensity={0.6} />
        <pointLight position={[10, 10, 10]} />
        <pointLight position={[-10, -10, -10]} intensity={0.3} />

        {filteredTokens.map((token, index) => {
          const position = [
            (token.embedding[0] - 0.5) * 6,
            (token.embedding[1] - 0.5) * 6,
            (token.embedding[2] - 0.5) * 6,
          ] as [number, number, number]

          return (
            <group key={index}>
              <Sphere
                position={position}
                args={[selectedToken === token.id ? 0.25 : 0.15, 32, 32]}
                onClick={() => onTokenSelect(token.id)}
              >
                <meshStandardMaterial
                  color={token.color}
                  emissive={selectedToken === token.id ? token.color : "#000000"}
                  emissiveIntensity={selectedToken === token.id ? 0.3 : 0}
                />
              </Sphere>

              <Text
                position={[position[0], position[1] + 0.4, position[2]]}
                fontSize={0.3}
                color={token.color}
                anchorX="center"
                anchorY="middle"
              >
                {token.text}
              </Text>

              {/* Enhanced relationship lines */}
              {showConnections && demoMode &&
                filteredTokens.map((otherToken: any, otherIndex: any) => {
                  if (otherIndex >= index) return null;
                  
                  // Check if these tokens have a semantic relationship
                  const relationship = SEMANTIC_RELATIONSHIPS.find(rel => 
                    (rel.token1 === token.text && rel.token2 === otherToken.text) ||
                    (rel.token2 === token.text && rel.token1 === otherToken.text)
                  );

                  if (!relationship) return null;

                  const otherPosition = [
                    (otherToken.embedding[0] - 0.5) * 6,
                    (otherToken.embedding[1] - 0.5) * 6,
                    (otherToken.embedding[2] - 0.5) * 6,
                  ] as [number, number, number];

                  return (
                    <Line
                      key={`connection-${index}-${otherIndex}`}
                      points={[position, otherPosition]}
                      color={relationship.color}
                      lineWidth={2}
                      transparent
                      opacity={0.8}
                    />
                  );
                })
              }

              {/* Standard similarity connections for non-demo mode */}
              {showConnections && !demoMode &&
                displayTokens.map((otherToken, otherIndex) => {
                  if (index !== otherIndex) {
                    const similarity = calculateSimilarity(token.embedding, otherToken.embedding)
                    if (similarity > 0.7) {
                      const otherPosition = [
                        (otherToken.embedding[0] - 0.5) * 6,
                        (otherToken.embedding[1] - 0.5) * 6,
                        (otherToken.embedding[2] - 0.5) * 6,
                      ] as [number, number, number]

                      return (
                        <Line
                          key={`${index}-${otherIndex}`}
                          points={[position, otherPosition]}
                          color="#888888"
                          lineWidth={similarity * 3}
                          transparent
                          opacity={0.4}
                        />
                      )
                    }
                  }
                  return null
                })}
            </group>
          )
        })}

        <OrbitControls enableZoom={true} />
      </Canvas>
    )
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center space-x-2">
              <Brain className="h-5 w-5" />
              <span>Token Embeddings Visualization</span>
              {demoMode && <Badge variant="secondary">Demo Mode</Badge>}
            </CardTitle>
            <div className="flex items-center space-x-2">
              <Button
                variant={demoMode ? "default" : "outline"}
                size="sm"
                onClick={() => setDemoMode(!demoMode)}
              >
                Demo Relationships
              </Button>
              <Button
                variant={showConnections ? "default" : "outline"}
                size="sm"
                onClick={() => setShowConnections(!showConnections)}
              >
                Connections
              </Button>
              <Button
                variant={dimension === "3d" ? "default" : "outline"}
                size="sm"
                onClick={() => setDimension(dimension === "3d" ? "2d" : "3d")}
              >
                {dimension === "3d" ? "3D" : "2D"}
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="visualization" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="visualization">3D Space</TabsTrigger>
              <TabsTrigger value="mathematics">Mathematics</TabsTrigger>
              <TabsTrigger value="analysis">Analysis</TabsTrigger>
            </TabsList>

            <TabsContent value="visualization" className="space-y-4">
              {demoMode && (
                <div className="p-4 bg-gradient-to-r from-purple-50 to-blue-50 dark:from-purple-900/20 dark:to-blue-900/20 rounded-lg">
                  <h4 className="font-medium mb-3">🎯 ChatGPT Processing: "Once upon a time in a magical forest"</h4>
                  <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                    Watch how ChatGPT understands this storytelling prompt internally. Each token is positioned in 3D space based on its meaning and relationship to other words.
                  </p>
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-2 mb-4">
                    {["all", "story_pattern", "setting", "location", "continuation"].map((relationship) => (
                      <Button
                        key={relationship}
                        variant={selectedRelationship === relationship ? "default" : "outline"}
                        size="sm"
                        onClick={() => setSelectedRelationship(relationship)}
                        className="text-xs"
                      >
                        {relationship === "all" ? "All" : 
                         relationship === "story_pattern" ? "� Story" :
                         relationship === "setting" ? "� Setting" :
                         relationship === "location" ? "� Location" :
                         "🔮 Next"}
                      </Button>
                    ))}
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                    <div className="p-3 bg-white dark:bg-gray-800 rounded">
                      <strong className="text-green-600">Story Pattern:</strong>
                      <div className="text-xs mt-1">"once upon a time" triggers narrative mode</div>
                      <div className="text-xs text-gray-500">Connected by green lines</div>
                    </div>
                    <div className="p-3 bg-white dark:bg-gray-800 rounded">
                      <strong className="text-purple-600">Setting:</strong>
                      <div className="text-xs mt-1">"magical forest" establishes fantasy world</div>
                      <div className="text-xs text-gray-500">Connected by purple lines</div>
                    </div>
                    <div className="p-3 bg-white dark:bg-gray-800 rounded">
                      <strong className="text-blue-600">Location:</strong>
                      <div className="text-xs mt-1">Spatial relationships and transitions</div>
                      <div className="text-xs text-gray-500">Connected by blue lines</div>
                    </div>
                    <div className="p-3 bg-white dark:bg-gray-800 rounded">
                      <strong className="text-red-600">Prediction:</strong>
                      <div className="text-xs mt-1">What the AI expects to generate next</div>
                      <div className="text-xs text-gray-500">Connected by red lines</div>
                    </div>
                  </div>
                </div>
              )}

              <div className="h-96 bg-gray-100 dark:bg-gray-700 rounded-lg">
                <Enhanced3DTokenCloud />
              </div>

              <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <h4 className="font-medium mb-2">Understanding Embeddings: From Words to Mathematical Vectors</h4>
                <div className="text-sm space-y-3">
                  <div className="p-3 bg-white dark:bg-gray-800 rounded border-l-4 border-purple-500">
                    <strong>What Are Vectors?</strong> Each token gets a unique vector - a list of 768+ floating-point numbers that captures everything the model learned about that token's meaning, usage, and relationships.
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                      <strong>Mathematical Operations:</strong>
                      <div className="text-xs mt-1 font-mono space-y-1">
                        <div>king - man + woman ≈ queen</div>
                        <div>Paris - France + Germany ≈ Berlin</div>
                        <div>walking - walk + swim ≈ swimming</div>
                      </div>
                    </div>
                    <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                      <strong>Similarity Calculation:</strong>
                      <div className="text-xs mt-1">
                        Cosine similarity ranges from -1 (opposite) to 1 (identical). Similar concepts cluster together in high-dimensional space.
                      </div>
                    </div>
                  </div>

                  <div className="p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded">
                    <strong>Why 768+ Dimensions?</strong> Higher dimensions allow more complex relationships. Different dimensions capture grammatical properties, semantic categories, emotional associations, and domain-specific knowledge without interference.
                  </div>

                  <div className="text-xs space-y-1">
                    <p><strong>Real embeddings have 768-4096 dimensions:</strong> We're showing a simplified 3D projection.</p>
                    <p><strong>Distance = Semantic similarity:</strong> Closer tokens have more similar meanings.</p>
                    <p><strong>Geometric Relationships:</strong> Analogies become vector arithmetic in embedding space.</p>
                  </div>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="mathematics" className="space-y-4">
              <div className="space-y-4">
                <Card className="p-4">
                  <h4 className="font-medium mb-3 flex items-center space-x-2">
                    <Calculator className="h-4 w-4" />
                    <span>Embedding Mathematics</span>
                  </h4>

                  <div className="space-y-4">
                    <div>
                      <h5 className="font-medium mb-2">1. Token to Vector Transformation</h5>
                      <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded font-mono text-sm">
                        embedding = E[token_id]<br/>
                        # E is a learned embedding matrix<br/>
                        # Shape: [vocab_size, embedding_dim]
                      </div>
                      <p className="text-sm text-gray-600 mt-2">
                        Each token ID maps to a learned vector of real numbers. GPT-4 uses ~100,000 × 12,288 = 1.2B embedding parameters.
                      </p>
                    </div>

                    <div>
                      <h5 className="font-medium mb-2">2. Cosine Similarity Formula</h5>
                      <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded font-mono text-sm">
                        similarity = (A · B) / (||A|| × ||B||)<br/>
                        = Σ(ai × bi) / (√Σai² × √Σbi²)
                      </div>
                      <p className="text-sm text-gray-600 mt-2">
                        Measures angle between vectors. Used for semantic similarity: cat ↔ dog = 0.73, cat ↔ car = 0.21
                      </p>
                    </div>

                    <div>
                      <h5 className="font-medium mb-2">3. Euclidean Distance</h5>
                      <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded font-mono text-sm">
                        distance = √Σ(ai - bi)²<br/>
                        # L2 norm in embedding space
                      </div>
                      <p className="text-sm text-gray-600 mt-2">
                        Straight-line distance between vectors in high-dimensional space. Closer = more semantically related.
                      </p>
                    </div>

                    <div>
                      <h5 className="font-medium mb-2">4. Vector Arithmetic (Analogies)</h5>
                      <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded font-mono text-sm">
                        Vector("king") - Vector("man") + Vector("woman")<br/>
                        ≈ Vector("queen")<br/>
                        # Semantic relationships become geometric
                      </div>
                      <p className="text-sm text-gray-600 mt-2">
                        Embeddings capture relational patterns: gender, geography, tense, etc. Linear operations on meaning!
                      </p>
                    </div>

                    <div>
                      <h5 className="font-medium mb-2">5. Learning Process</h5>
                      <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded font-mono text-sm">
                        # Initial: Random vectors<br/>
                        # Training: Adjust via gradient descent<br/>
                        # Objective: Predict context words<br/>
                        # Result: Semantic structure emerges
                      </div>
                      <p className="text-sm text-gray-600 mt-2">
                        Embeddings start random, then develop meaning through training on billions of text examples.
                      </p>
                    </div>
                  </div>
                </Card>

                {selectedToken !== null && (
                  <Card className="p-4">
                    <h4 className="font-medium mb-3">Selected Token Mathematics</h4>
                    {(() => {
                      const token = tokens.find((t) => t.id === selectedToken)
                      return token ? (
                        <div className="space-y-3">
                          <div>
                            <h5 className="font-medium mb-2">Embedding Vector for "{token.text}" (simplified 5D):</h5>
                            <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded font-mono text-sm">
                              [{token.embedding.map((val) => val.toFixed(3)).join(", ")}]
                            </div>
                            <div className="text-xs text-gray-500 mt-1">
                              Real models use 768-12,288 dimensions. Each number encodes semantic properties.
                            </div>
                          </div>

                          <div>
                            <h5 className="font-medium mb-2">Vector Properties:</h5>
                            <div className="grid grid-cols-2 gap-3">
                              <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded">
                                <div className="text-xs text-gray-600 dark:text-gray-300">Magnitude (||v||)</div>
                                <div className="font-mono text-sm">{Math.sqrt(token.embedding.reduce((sum, val) => sum + val * val, 0)).toFixed(3)}</div>
                              </div>
                              <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded">
                                <div className="text-xs text-gray-600 dark:text-gray-300">Unit Vector</div>
                                <div className="font-mono text-xs">
                                  [{token.embedding.map(val => (val / Math.sqrt(token.embedding.reduce((sum, v) => sum + v * v, 0))).toFixed(2)).join(", ")}]
                                </div>
                              </div>
                            </div>
                          </div>

                          <div>
                            <h5 className="font-medium mb-2">Dimensional Analysis (Hypothetical):</h5>
                            <div className="space-y-2 text-xs">
                              <div className="flex justify-between">
                                <span>Dim 1 (Grammatical): {token.embedding[0] > 0 ? 'Noun-like' : 'Verb-like'}</span>
                                <span className="font-mono">{token.embedding[0].toFixed(3)}</span>
                              </div>
                              <div className="flex justify-between">
                                <span>Dim 2 (Semantic): {token.embedding[1] > 0 ? 'Concrete' : 'Abstract'}</span>
                                <span className="font-mono">{token.embedding[1].toFixed(3)}</span>
                              </div>
                              <div className="flex justify-between">
                                <span>Dim 3 (Emotional): {token.embedding[2] > 0 ? 'Positive' : 'Negative'}</span>
                                <span className="font-mono">{token.embedding[2].toFixed(3)}</span>
                              </div>
                            </div>
                          </div>

                          <div>
                            <h5 className="font-medium mb-2">Similarities to other tokens:</h5>
                            <div className="space-y-1 max-h-32 overflow-y-auto">
                              {tokens
                                .filter((t) => t.id !== token.id)
                                .map((otherToken) => {
                                  const similarity = calculateSimilarity(token.embedding, otherToken.embedding)
                                  return (
                                    <div key={otherToken.id} className="flex justify-between text-sm">
                                      <span className="flex items-center space-x-2">
                                        <div className="w-2 h-2 rounded-full" style={{backgroundColor: otherToken.color}}/>
                                        <span>"{otherToken.text}"</span>
                                      </span>
                                      <div className="flex items-center space-x-2">
                                        <span className="font-mono">{similarity.toFixed(3)}</span>
                                        <span className="text-xs">
                                          {similarity > 0.8 ? '🔥' : similarity > 0.6 ? '👀' : similarity > 0.4 ? '🤔' : '💭'}
                                        </span>
                                      </div>
                                    </div>
                                  )
                                })}
                            </div>
                          </div>
                        </div>
                      ) : null
                    })()}
                  </Card>
                )}
              </div>
            </TabsContent>

            <TabsContent value="analysis" className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card className="p-4">
                  <h4 className="font-medium mb-3">Embedding Statistics</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span>Dimensions:</span>
                      <Badge>5 (simplified)</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span>Real Model Dims:</span>
                      <Badge>768-4096</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span>Vocabulary Size:</span>
                      <Badge>50K-100K+</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span>Total Parameters:</span>
                      <Badge>Millions</Badge>
                    </div>
                  </div>
                </Card>

                <Card className="p-4">
                  <h4 className="font-medium mb-3">Clustering Analysis</h4>
                  <div className="space-y-3">
                    {tokens.reduce((clusters: { [key: string]: TokenData[] }, token) => {
                      if (!clusters[token.semantic_role]) {
                        clusters[token.semantic_role] = []
                      }
                      clusters[token.semantic_role].push(token)
                      return clusters
                    }, {}) &&
                      Object.entries(
                        tokens.reduce((clusters: { [key: string]: TokenData[] }, token) => {
                          if (!clusters[token.semantic_role]) {
                            clusters[token.semantic_role] = []
                          }
                          clusters[token.semantic_role].push(token)
                          return clusters
                        }, {}),
                      ).map(([role, roleTokens]) => (
                        <div key={role} className="p-2 bg-gray-50 dark:bg-gray-800 rounded">
                          <div className="font-medium text-sm">{role}</div>
                          <div className="text-xs text-gray-600 dark:text-gray-300">
                            {roleTokens.map((t) => t.text).join(", ")}
                          </div>
                        </div>
                      ))}
                  </div>
                </Card>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {/* Token Legend */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Target className="h-5 w-5" />
            <span>Token Legend & Details</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {tokens.map((token) => (
              <motion.div
                key={token.id}
                className={`flex items-center space-x-3 p-3 rounded cursor-pointer transition-all ${
                  selectedToken === token.id
                    ? "bg-blue-100 dark:bg-blue-900 ring-2 ring-blue-500"
                    : "hover:bg-gray-100 dark:hover:bg-gray-700"
                }`}
                onClick={() => onTokenSelect(selectedToken === token.id ? null : token.id)}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <div className="w-4 h-4 rounded-full" style={{ backgroundColor: token.color }} />
                <div className="flex-1 min-w-0">
                  <div className="font-medium truncate">{token.text}</div>
                  <div className="text-xs text-gray-500 truncate">{token.semantic_role}</div>
                </div>
                <div className="text-xs font-mono text-gray-400">{token.embedding[0].toFixed(2)}</div>
              </motion.div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
