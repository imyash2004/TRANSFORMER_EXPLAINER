"use client"

import { useState } from "react"
import { Canvas } from "@react-three/fiber"
import { OrbitControls, Text, Sphere, Line } from "@react-three/drei"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Brain, Calculator, Target } from "lucide-react"
import type { TokenData } from "@/lib/textAnalyzer"
import { motion } from "framer-motion"

interface EmbeddingVisualizationProps {
  tokens: TokenData[]
  selectedToken: number | null
  onTokenSelect: (tokenId: number | null) => void
}

export function EmbeddingVisualization({ tokens, selectedToken, onTokenSelect }: EmbeddingVisualizationProps) {
  const [showConnections, setShowConnections] = useState(true)
  const [dimension, setDimension] = useState<"3d" | "2d">("3d")

  // Calculate similarity between tokens
  const calculateSimilarity = (embedding1: number[], embedding2: number[]): number => {
    const dotProduct = embedding1.reduce((sum, val, i) => sum + val * embedding2[i], 0)
    const magnitude1 = Math.sqrt(embedding1.reduce((sum, val) => sum + val * val, 0))
    const magnitude2 = Math.sqrt(embedding2.reduce((sum, val) => sum + val * val, 0))
    return dotProduct / (magnitude1 * magnitude2)
  }

  // Enhanced 3D visualization component
  function Enhanced3DTokenCloud() {
    return (
      <Canvas camera={{ position: [0, 0, 8] }}>
        <ambientLight intensity={0.6} />
        <pointLight position={[10, 10, 10]} />
        <pointLight position={[-10, -10, -10]} intensity={0.3} />

        {tokens.map((token, index) => {
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

              {/* Connection lines to similar tokens */}
              {showConnections &&
                tokens.map((otherToken, otherIndex) => {
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
            </CardTitle>
            <div className="flex items-center space-x-2">
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
              <div className="h-96 bg-gray-100 dark:bg-gray-700 rounded-lg">
                <Enhanced3DTokenCloud />
              </div>

              <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <h4 className="font-medium mb-2">Understanding Embeddings</h4>
                <div className="text-sm space-y-2">
                  <p>
                    <strong>What you're seeing:</strong> Each token is positioned in 3D space based on its meaning.
                  </p>
                  <p>
                    <strong>Similar tokens cluster together:</strong> Words with similar meanings are closer in space.
                  </p>
                  <p>
                    <strong>Distance = Semantic similarity:</strong> The closer two tokens, the more similar their
                    meanings.
                  </p>
                  <p>
                    <strong>Real embeddings have 768+ dimensions:</strong> We're showing a simplified 3D projection.
                  </p>
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
                      <h5 className="font-medium mb-2">1. Token to Vector Conversion</h5>
                      <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded font-mono text-sm">
                        embedding = E[token_id]
                      </div>
                      <p className="text-sm text-gray-600 mt-2">
                        Each token ID maps to a learned vector of real numbers
                      </p>
                    </div>

                    <div>
                      <h5 className="font-medium mb-2">2. Similarity Calculation (Cosine Similarity)</h5>
                      <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded font-mono text-sm">
                        similarity = (A · B) / (||A|| × ||B||)
                      </div>
                      <p className="text-sm text-gray-600 mt-2">Where A and B are embedding vectors</p>
                    </div>

                    <div>
                      <h5 className="font-medium mb-2">3. Distance in Embedding Space</h5>
                      <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded font-mono text-sm">
                        distance = √Σ(ai - bi)²
                      </div>
                      <p className="text-sm text-gray-600 mt-2">Euclidean distance between two embedding vectors</p>
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
                            <h5 className="font-medium mb-2">Embedding Vector for "{token.text}":</h5>
                            <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded font-mono text-sm">
                              [{token.embedding.map((val) => val.toFixed(3)).join(", ")}]
                            </div>
                          </div>

                          <div>
                            <h5 className="font-medium mb-2">Vector Magnitude:</h5>
                            <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded font-mono text-sm">
                              ||v|| = {Math.sqrt(token.embedding.reduce((sum, val) => sum + val * val, 0)).toFixed(3)}
                            </div>
                          </div>

                          <div>
                            <h5 className="font-medium mb-2">Similarities to other tokens:</h5>
                            <div className="space-y-1">
                              {tokens
                                .filter((t) => t.id !== token.id)
                                .map((otherToken) => {
                                  const similarity = calculateSimilarity(token.embedding, otherToken.embedding)
                                  return (
                                    <div key={otherToken.id} className="flex justify-between text-sm">
                                      <span>"{otherToken.text}":</span>
                                      <span className="font-mono">{similarity.toFixed(3)}</span>
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
