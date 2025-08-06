"use client"

import { motion } from "framer-motion"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { Play, Pause, RotateCcw, Info } from "lucide-react"
import type { TokenData } from "@/lib/textAnalyzer"

interface TokenVisualizationProps {
  tokens: TokenData[]
  selectedToken: number | null
  onTokenSelect: (tokenId: number | null) => void
  currentStep: number
  isPlaying: boolean
  onPlayToggle: () => void
  onReset: () => void
}

export function TokenVisualization({
  tokens,
  selectedToken,
  onTokenSelect,
  currentStep,
  isPlaying,
  onPlayToggle,
  onReset,
}: TokenVisualizationProps) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center space-x-2">
            <span>Interactive Tokenization</span>
            <Badge variant="outline">{tokens.length} tokens</Badge>
          </CardTitle>
          <div className="flex items-center space-x-2">
            <Button variant="outline" size="sm" onClick={onPlayToggle}>
              {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              {isPlaying ? "Pause" : "Play"}
            </Button>
            <Button variant="outline" size="sm" onClick={onReset}>
              <RotateCcw className="h-4 w-4" />
              Reset
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Step-by-step explanation */}
        <div className="space-y-4">
          <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
            <h4 className="font-medium mb-2 flex items-center space-x-2">
              <Info className="h-4 w-4" />
              <span>The Fundamental Question: Are Tokens Equal to Words?</span>
            </h4>
            <div className="text-sm space-y-3">
              <div className="p-3 bg-white dark:bg-gray-800 rounded border-l-4 border-yellow-500">
                <strong>The Short Answer:</strong> Tokens are NOT words. They are the fundamental atomic units that language models use to process text, representing words, parts of words, punctuation, or special symbols.
              </div>
              
              <div className="space-y-2">
                <div className="flex items-start space-x-2">
                  <div className="w-6 h-6 rounded-full bg-blue-500 text-white text-xs flex items-center justify-center mt-0.5">1</div>
                  <div>
                    <strong>Whole Words:</strong> Common words get their own tokens
                    <div className="text-xs text-gray-600 mt-1 font-mono">"the" → Token ID: 464</div>
                  </div>
                </div>
                <div className="flex items-start space-x-2">
                  <div className="w-6 h-6 rounded-full bg-blue-500 text-white text-xs flex items-center justify-center mt-0.5">2</div>
                  <div>
                    <strong>Subwords:</strong> Complex words are broken into pieces
                    <div className="text-xs text-gray-600 mt-1 font-mono">"unhappiness" → ["un", "happy", "ness"]</div>
                  </div>
                </div>
                <div className="flex items-start space-x-2">
                  <div className="w-6 h-6 rounded-full bg-blue-500 text-white text-xs flex items-center justify-center mt-0.5">3</div>
                  <div>
                    <strong>Byte Pair Encoding (BPE):</strong> Algorithm merges frequent character pairs
                    <div className="text-xs text-gray-600 mt-1">Most frequent pair "th" → single token</div>
                  </div>
                </div>
                <div className="flex items-start space-x-2">
                  <div className="w-6 h-6 rounded-full bg-blue-500 text-white text-xs flex items-center justify-center mt-0.5">4</div>
                  <div>
                    <strong>Special Tokens:</strong> Control symbols for model operation
                    <div className="text-xs text-gray-600 mt-1 font-mono">&lt;pad&gt;, &lt;unk&gt;, &lt;s&gt;, &lt;/s&gt;</div>
                  </div>
                </div>
              </div>

              <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                <strong>Why This Matters:</strong> Tokenization affects model performance, handling of rare words, multilingual capability, and computational efficiency. Modern models like GPT-4 use ~100,000 tokens in their vocabulary.
              </div>
            </div>
          </div>

          {/* Animated tokens */}
          <div>
            <h4 className="font-medium mb-3">Tokenized Output:</h4>
            <div className="flex flex-wrap gap-3 mb-4">
              {tokens.map((token, index) => (
                <TooltipProvider key={token.id}>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <motion.div
                        initial={{ opacity: 0, scale: 0, y: 20 }}
                        animate={{
                          opacity: index <= currentStep ? 1 : 0.3,
                          scale: index <= currentStep ? 1 : 0.8,
                          y: 0,
                        }}
                        transition={{ delay: index * 0.2 }}
                        className={`relative cursor-pointer transition-all duration-200 ${
                          selectedToken === token.id ? "scale-110 ring-2 ring-white" : "hover:scale-105"
                        }`}
                        onClick={() => onTokenSelect(selectedToken === token.id ? null : token.id)}
                      >
                        <div
                          className="px-4 py-3 rounded-lg text-white font-medium shadow-lg"
                          style={{ backgroundColor: token.color }}
                        >
                          <div className="text-center">
                            <div className="font-mono text-sm">{token.text}</div>
                            <div className="text-xs opacity-75 mt-1">ID: {token.id}</div>
                            <div className="text-xs opacity-75">{token.pos_tag}</div>
                          </div>
                        </div>
                        {index <= currentStep && (
                          <motion.div
                            initial={{ scale: 0 }}
                            animate={{ scale: 1 }}
                            className="absolute -top-2 -right-2 bg-green-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-xs"
                          >
                            ✓
                          </motion.div>
                        )}
                      </motion.div>
                    </TooltipTrigger>
                    <TooltipContent className="max-w-xs">
                      <div className="space-y-2">
                        <div className="font-semibold">Token: "{token.text}"</div>
                        <div className="text-sm">POS: {token.pos_tag}</div>
                        <div className="text-sm">Role: {token.semantic_role}</div>
                        <div className="text-sm">Frequency: {(token.frequency * 100).toFixed(2)}%</div>
                      </div>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              ))}
            </div>
          </div>

          {/* Selected token details */}
          {selectedToken !== null && (
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
              <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                {(() => {
                  const token = tokens.find((t) => t.id === selectedToken)
                  return token ? (
                    <div className="space-y-3">
                      <div className="flex items-center space-x-3">
                        <div className="w-6 h-6 rounded-full" style={{ backgroundColor: token.color }} />
                        <h4 className="font-semibold text-lg">Token Deep Dive: "{token.text}"</h4>
                      </div>

                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="text-center p-3 bg-white dark:bg-gray-700 rounded">
                          <div className="text-sm text-gray-600 dark:text-gray-300">Token ID</div>
                          <div className="font-bold text-lg">{token.id}</div>
                          <div className="text-xs text-gray-500">Vocabulary Index</div>
                        </div>
                        <div className="text-center p-3 bg-white dark:bg-gray-700 rounded">
                          <div className="text-sm text-gray-600 dark:text-gray-300">POS Tag</div>
                          <div className="font-bold text-lg">{token.pos_tag}</div>
                          <div className="text-xs text-gray-500">Grammar Role</div>
                        </div>
                        <div className="text-center p-3 bg-white dark:bg-gray-700 rounded">
                          <div className="text-sm text-gray-600 dark:text-gray-300">Frequency</div>
                          <div className="font-bold text-lg">{(token.frequency * 100).toFixed(1)}%</div>
                          <div className="text-xs text-gray-500">Training Corpus</div>
                        </div>
                        <div className="text-center p-3 bg-white dark:bg-gray-700 rounded">
                          <div className="text-sm text-gray-600 dark:text-gray-300">Sentiment</div>
                          <div className="font-bold text-lg">
                            {token.sentiment > 0 ? "😊" : token.sentiment < 0 ? "😔" : "😐"}
                          </div>
                          <div className="text-xs text-gray-500">Emotional Valence</div>
                        </div>
                      </div>

                      <div className="space-y-3">
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                          <div className="text-sm font-medium mb-1">Tokenization Analysis:</div>
                          <div className="text-sm text-gray-700 dark:text-gray-300">{token.explanation}</div>
                        </div>

                        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                          <div className="text-sm font-medium mb-2">BPE Process for "{token.text}":</div>
                          <div className="text-xs space-y-1">
                            <div>1. Character-level: {token.text.split('').map(c => `"${c}"`).join(', ')}</div>
                            <div>2. Frequency analysis across training corpus</div>
                            <div>3. Merge most frequent character pairs</div>
                            <div>4. Result: Single token with ID {token.id}</div>
                          </div>
                        </div>

                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                          <div className="text-sm font-medium mb-2">Vocabulary Insights:</div>
                          <div className="text-xs space-y-1">
                            <div>• Modern LLMs use 50K-100K+ token vocabularies</div>
                            <div>• This token represents {((token.frequency) * 100).toFixed(3)}% of training data</div>
                            <div>• Subword tokenization enables handling of rare/new words</div>
                            <div>• GPT-4 uses ~100,000 tokens for optimal efficiency</div>
                          </div>
                        </div>
                      </div>
                    </div>
                  ) : null
                })()}
              </div>
            </motion.div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
