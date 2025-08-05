"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { Network, Calculator, Info, Eye } from "lucide-react"
import type { TokenData } from "@/lib/textAnalyzer"
import { motion } from "framer-motion"

interface AttentionVisualizationProps {
  tokens: TokenData[]
  selectedToken: number | null
  onTokenSelect: (tokenId: number | null) => void
}

// Generate attention matrices for visualization
function generateAttentionMatrix(tokens: TokenData[], layer: number, head: number): number[][] {
  const size = tokens.length
  const matrix: number[][] = []

  for (let i = 0; i < size; i++) {
    matrix[i] = []
    for (let j = 0; j < size; j++) {
      if (i === j) {
        // Self-attention is usually high
        matrix[i][j] = 0.7 + Math.random() * 0.25
      } else {
        // Calculate attention based on token similarity and position
        const positionWeight = 1 / (Math.abs(i - j) + 1)
        const semanticSimilarity = calculateTokenSimilarity(tokens[i], tokens[j])
        const headBias = Math.sin((layer + head + i + j) * 0.5) * 0.2

        matrix[i][j] = Math.max(
          0.01,
          Math.min(0.95, positionWeight * 0.3 + semanticSimilarity * 0.5 + headBias + Math.random() * 0.2),
        )
      }
    }

    // Normalize row to sum to 1 (softmax-like)
    const rowSum = matrix[i].reduce((sum, val) => sum + val, 0)
    matrix[i] = matrix[i].map((val) => val / rowSum)
  }

  return matrix
}

function calculateTokenSimilarity(token1: TokenData, token2: TokenData): number {
  // Simple similarity based on semantic role and POS tag
  let similarity = 0

  if (token1.semantic_role === token2.semantic_role) similarity += 0.4
  if (token1.pos_tag === token2.pos_tag) similarity += 0.3

  // Add embedding similarity
  const embeddingSim =
    token1.embedding.reduce((sum, val, i) => sum + val * token2.embedding[i], 0) /
    (Math.sqrt(token1.embedding.reduce((sum, val) => sum + val * val, 0)) *
      Math.sqrt(token2.embedding.reduce((sum, val) => sum + val * val, 0)))

  similarity += embeddingSim * 0.3

  return Math.max(0, Math.min(1, similarity))
}

// Attention heatmap component
function AttentionHeatmap({
  matrix,
  tokens,
  onCellHover,
  selectedCell,
}: {
  matrix: number[][]
  tokens: TokenData[]
  onCellHover: (i: number, j: number, weight: number) => void
  selectedCell: { i: number; j: number } | null
}) {
  return (
    <div className="space-y-2">
      <div className="grid gap-1 text-xs" style={{ gridTemplateColumns: `auto repeat(${tokens.length}, 1fr)` }}>
        <div></div>
        {tokens.map((token, i) => (
          <div key={i} className="text-center font-medium p-1 truncate" style={{ color: token.color }}>
            {token.text}
          </div>
        ))}
      </div>

      {matrix.map((row, i) => (
        <div key={i} className="grid gap-1" style={{ gridTemplateColumns: `auto repeat(${tokens.length}, 1fr)` }}>
          <div className="text-xs font-medium p-1 text-right truncate" style={{ color: tokens[i]?.color }}>
            {tokens[i]?.text}
          </div>
          {row.map((weight, j) => (
            <TooltipProvider key={j}>
              <Tooltip>
                <TooltipTrigger asChild>
                  <motion.div
                    className={`h-8 rounded cursor-pointer transition-all ${
                      selectedCell?.i === i && selectedCell?.j === j ? "ring-2 ring-blue-500" : ""
                    }`}
                    style={{
                      backgroundColor: `rgba(59, 130, 246, ${weight})`,
                      border: weight > 0.7 ? "2px solid #3B82F6" : "1px solid rgba(0,0,0,0.1)",
                    }}
                    onMouseEnter={() => onCellHover(i, j, weight)}
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.95 }}
                  />
                </TooltipTrigger>
                <TooltipContent>
                  <div className="space-y-1">
                    <p>
                      <strong>From:</strong> "{tokens[i]?.text}"
                    </p>
                    <p>
                      <strong>To:</strong> "{tokens[j]?.text}"
                    </p>
                    <p>
                      <strong>Attention:</strong> {(weight * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs text-gray-500">
                      {weight > 0.7 ? "Strong attention" : weight > 0.4 ? "Moderate attention" : "Weak attention"}
                    </p>
                  </div>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          ))}
        </div>
      ))}
    </div>
  )
}

export function AttentionVisualization({ tokens, selectedToken, onTokenSelect }: AttentionVisualizationProps) {
  const [currentLayer, setCurrentLayer] = useState([1])
  const [selectedHead, setSelectedHead] = useState(0)
  const [hoveredCell, setHoveredCell] = useState<{ i: number; j: number; weight: number } | null>(null)
  const [selectedCell, setSelectedCell] = useState<{ i: number; j: number } | null>(null)

  const attentionMatrix = generateAttentionMatrix(tokens, currentLayer[0], selectedHead)

  const headExplanations = [
    "Syntactic relationships - connects verbs with their objects and subjects",
    "Semantic similarity - links words with related meanings",
    "Positional relationships - focuses on word order and sequence",
    "Grammatical dependencies - handles subject-verb agreement",
    "Long-range dependencies - connects distant but related words",
    "Local context - focuses on immediate neighboring words",
    "Entity relationships - connects proper nouns and related concepts",
    "Modifier relationships - links adjectives to nouns they modify",
    "Temporal relationships - connects time-related expressions",
    "Causal relationships - identifies cause and effect patterns",
    "Comparative relationships - handles comparisons and contrasts",
    "Discourse relationships - manages topic flow and coherence",
  ]

  return (
    <div className="space-y-6">
      {/* Controls */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span className="flex items-center space-x-2">
              <Network className="h-5 w-5" />
              <span>Multi-Head Attention Controls</span>
            </span>
            <Badge variant="outline">
              Layer {currentLayer[0]} • Head {selectedHead + 1}
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">Transformer Layer</label>
                <span className="text-sm font-mono">{currentLayer[0]}/12</span>
              </div>
              <Slider
                value={currentLayer}
                onValueChange={setCurrentLayer}
                max={12}
                min={1}
                step={1}
                className="w-full"
              />
              <div className="text-xs text-gray-600 dark:text-gray-300">
                Each layer learns different types of relationships
              </div>
            </div>

            <div className="space-y-3">
              <div className="text-sm font-medium">Attention Head</div>
              <div className="grid grid-cols-6 gap-2">
                {Array.from({ length: 12 }, (_, i) => (
                  <Button
                    key={i}
                    variant={selectedHead === i ? "default" : "outline"}
                    size="sm"
                    onClick={() => setSelectedHead(i)}
                    className="h-8 text-xs"
                  >
                    {i + 1}
                  </Button>
                ))}
              </div>
              <div className="text-xs text-gray-600 dark:text-gray-300">
                Each head specializes in different patterns
              </div>
            </div>
          </div>

          <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
            <div className="flex items-start space-x-2">
              <Info className="h-4 w-4 mt-0.5 text-blue-600" />
              <div>
                <div className="text-sm font-medium text-blue-800 dark:text-blue-200">
                  Head {selectedHead + 1} Focus:
                </div>
                <div className="text-sm text-blue-700 dark:text-blue-300">{headExplanations[selectedHead]}</div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Attention Visualization */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">
                Attention Matrix - Layer {currentLayer[0]}, Head {selectedHead + 1}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <AttentionHeatmap
                matrix={attentionMatrix}
                tokens={tokens}
                onCellHover={(i, j, weight) => {
                  setHoveredCell({ i, j, weight })
                  setSelectedCell({ i, j })
                }}
                selectedCell={selectedCell}
              />

              <div className="mt-4 p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div className="text-sm space-y-1">
                  <div className="flex items-center space-x-2">
                    <div className="w-4 h-4 bg-blue-200 rounded"></div>
                    <span>Low attention (0-30%)</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-4 h-4 bg-blue-500 rounded"></div>
                    <span>Medium attention (30-70%)</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-4 h-4 bg-blue-800 rounded"></div>
                    <span>High attention (70-100%)</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center space-x-2">
                <Calculator className="h-4 w-4" />
                <span>Mathematics</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <h4 className="font-medium mb-2">Attention Formula:</h4>
                <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded font-mono text-sm">
                  Attention(Q,K,V) = softmax(QK^T/√d_k)V
                </div>
              </div>

              <div className="space-y-2">
                <div>
                  <strong>Q</strong> = Query matrix
                </div>
                <div>
                  <strong>K</strong> = Key matrix
                </div>
                <div>
                  <strong>V</strong> = Value matrix
                </div>
                <div>
                  <strong>d_k</strong> = Key dimension (64)
                </div>
              </div>

              <div>
                <h4 className="font-medium mb-2">Step-by-step:</h4>
                <div className="text-sm space-y-1">
                  <div>1. Compute Q·K^T (dot products)</div>
                  <div>2. Scale by √d_k</div>
                  <div>3. Apply softmax</div>
                  <div>4. Multiply by V</div>
                </div>
              </div>
            </CardContent>
          </Card>

          {hoveredCell && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center space-x-2">
                  <Eye className="h-4 w-4" />
                  <span>Attention Details</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div>
                    <div className="text-sm text-gray-600 dark:text-gray-300">From Token:</div>
                    <div className="font-medium flex items-center space-x-2">
                      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: tokens[hoveredCell.i]?.color }} />
                      <span>"{tokens[hoveredCell.i]?.text}"</span>
                    </div>
                  </div>

                  <div>
                    <div className="text-sm text-gray-600 dark:text-gray-300">To Token:</div>
                    <div className="font-medium flex items-center space-x-2">
                      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: tokens[hoveredCell.j]?.color }} />
                      <span>"{tokens[hoveredCell.j]?.text}"</span>
                    </div>
                  </div>

                  <div>
                    <div className="text-sm text-gray-600 dark:text-gray-300">Attention Weight:</div>
                    <div className="font-bold text-lg">{(hoveredCell.weight * 100).toFixed(1)}%</div>
                  </div>

                  <div className="p-2 bg-blue-50 dark:bg-blue-900/20 rounded text-sm">
                    {hoveredCell.weight > 0.7
                      ? "🔥 Strong attention - this token is very important for understanding the other"
                      : hoveredCell.weight > 0.4
                        ? "👀 Moderate attention - some relationship exists"
                        : "💭 Weak attention - minimal direct relationship"}
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Pattern Analysis</CardTitle>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="patterns">
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="patterns">Patterns</TabsTrigger>
                  <TabsTrigger value="stats">Statistics</TabsTrigger>
                </TabsList>

                <TabsContent value="patterns" className="space-y-3">
                  <div className="space-y-2">
                    <div className="p-2 bg-green-50 dark:bg-green-900/20 rounded text-sm">
                      <div className="font-medium">Self-Attention</div>
                      <div className="text-xs text-gray-600 dark:text-gray-300">
                        Avg:{" "}
                        {((attentionMatrix.reduce((sum, row, i) => sum + row[i], 0) / tokens.length) * 100).toFixed(1)}%
                      </div>
                    </div>

                    <div className="p-2 bg-blue-50 dark:bg-blue-900/20 rounded text-sm">
                      <div className="font-medium">Adjacent Attention</div>
                      <div className="text-xs text-gray-600 dark:text-gray-300">Tokens often attend to neighbors</div>
                    </div>

                    <div className="p-2 bg-purple-50 dark:bg-purple-900/20 rounded text-sm">
                      <div className="font-medium">Long-Range</div>
                      <div className="text-xs text-gray-600 dark:text-gray-300">Some distant connections exist</div>
                    </div>
                  </div>
                </TabsContent>

                <TabsContent value="stats" className="space-y-3">
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Max Attention:</span>
                      <span className="font-mono">{(Math.max(...attentionMatrix.flat()) * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Min Attention:</span>
                      <span className="font-mono">{(Math.min(...attentionMatrix.flat()) * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Avg Attention:</span>
                      <span className="font-mono">
                        {(
                          (attentionMatrix.flat().reduce((a, b) => a + b, 0) / attentionMatrix.flat().length) *
                          100
                        ).toFixed(1)}
                        %
                      </span>
                    </div>
                  </div>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
