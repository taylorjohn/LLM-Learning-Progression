import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';

const InnerThinkingTransformerGuide = () => {
  const [activeTab, setActiveTab] = useState('overview');
  
  return (
    <div className="w-full max-w-6xl mx-auto p-4">
      <h1 className="text-3xl font-bold mb-6">Inner Thinking Transformer (ITT)</h1>
      <p className="text-lg mb-8">
        An efficient transformer architecture that dynamically allocates computation to tokens
        based on their complexity, achieving better performance with fewer parameters.
      </p>
      
      {/* Navigation Tabs */}
      <div className="flex space-x-4 mb-8">
        {['overview', 'adaptive-routing', 'residual-thinking', 'comparison'].map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 rounded ${
              activeTab === tab 
                ? 'bg-blue-500 text-white' 
                : 'bg-gray-200 hover:bg-gray-300'
            }`}
          >
            {tab.split('-').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
          </button>
        ))}
      </div>
      
      {/* Overview Tab */}
      {activeTab === 'overview' && (
        <Card>
          <CardHeader>
            <CardTitle>ITT Architecture Overview</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div>
                <h3 className="font-bold text-lg mb-4">Key Innovations</h3>
                <ul className="list-disc ml-6 space-y-2">
                  <li className="text-blue-600 font-medium">Adaptive Token Routing</li>
                  <li className="text-green-600 font-medium">Residual Thinking Connections</li>
                  <li className="text-purple-600 font-medium">Computation Focusing</li>
                  <li className="text-amber-600 font-medium">Depth-Adaptive Processing</li>
                </ul>
                
                <h3 className="font-bold text-lg mt-6 mb-4">Benefits</h3>
                <ul className="list-disc ml-6 space-y-2">
                  <li>Smaller model size (162M vs 466M parameters)</li>
                  <li>Better performance on complex reasoning tasks</li>
                  <li>More efficient computation allocation</li>
                  <li>Reduced training and inference costs</li>
                </ul>
              </div>
              
              <div className="bg-gray-50 p-4 rounded">
                <h3 className="font-bold text-lg mb-4 text-center">Architecture Diagram</h3>
                <div className="space-y-3">
                  <div className="bg-blue-100 p-3 rounded text-center">Input Embeddings</div>
                  <div className="text-center">↓</div>
                  
                  <div className="bg-purple-100 p-3 rounded text-center font-bold">
                    Token Complexity Estimation
                  </div>
                  <div className="text-center">↓</div>
                  
                  <div className="grid grid-cols-3 gap-2">
                    <div className="bg-red-100 p-2 rounded text-center text-sm">
                      Low Complexity
                      <div className="text-xs mt-1">1 Thinking Step</div>
                    </div>
                    <div className="bg-yellow-100 p-2 rounded text-center text-sm">
                      Medium Complexity
                      <div className="text-xs mt-1">2 Thinking Steps</div>
                    </div>
                    <div className="bg-green-100 p-2 rounded text-center text-sm">
                      High Complexity
                      <div className="text-xs mt-1">4 Thinking Steps</div>
                    </div>
                  </div>
                  <div className="text-center">↓</div>
                  
                  <div className="bg-blue-100 p-3 rounded text-center">
                    Transformer Layers with Thinking Blocks
                  </div>
                  <div className="text-center">↓</div>
                  
                  <div className="bg-green-100 p-3 rounded text-center">
                    Output Predictions
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
      
      {/* Adaptive Routing Tab */}
      {activeTab === 'adaptive-routing' && (
        <Card>
          <CardHeader>
            <CardTitle>Adaptive Token Routing</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div>
                <h3 className="font-bold text-lg mb-4">How It Works</h3>
                <p className="mb-4">
                  Adaptive Token Routing dynamically allocates computational resources based on token complexity.
                  Complex tokens (those requiring deep reasoning) receive more thinking steps, while simple tokens
                  receive fewer steps.
                </p>
                
                <div className="bg-yellow-50 p-4 rounded mb-4">
                  <h4 className="font-bold mb-2">Complexity Estimation</h4>
                  <p>
                    A neural network estimates each token's complexity based on its embedding
                    and contextual information, assigning it 1-4 thinking steps.
                  </p>
                </div>
                
                <div className="bg-blue-50 p-4 rounded">
                  <h4 className="font-bold mb-2">Implementation Pseudocode</h4>
                  <pre className="text-xs overflow-x-auto">
{`# Estimate token complexity
complexity = sigmoid(linear_layer(token_embedding))

# Determine thinking steps (1-4)
thinking_steps = floor(complexity * max_steps) + 1

# Process only tokens that need extra thinking
for step in range(1, max_steps + 1):
    active_tokens = (thinking_steps >= step)
    if active_tokens.any():
        # Apply thinking only to active tokens
        tokens[active_tokens] = thinking_block(tokens[active_tokens])`}
                  </pre>
                </div>
              </div>
              
              <div>
                <h3 className="font-bold text-lg mb-4 text-center">Visualization</h3>
                <div className="bg-gray-50 rounded p-4">
                  <div className="text-center mb-2">Example Sentence Tokens</div>
                  <div className="grid grid-cols-8 gap-1">
                    <div className="p-2 bg-red-100 rounded text-center text-xs">The</div>
                    <div className="p-2 bg-red-100 rounded text-center text-xs">cat</div>
                    <div className="p-2 bg-yellow-100 rounded text-center text-xs">jumped</div>
                    <div className="p-2 bg-red-100 rounded text-center text-xs">over</div>
                    <div className="p-2 bg-red-100 rounded text-center text-xs">the</div>
                    <div className="p-2 bg-yellow-100 rounded text-center text-xs">lazy</div>
                    <div className="p-2 bg-green-100 rounded text-center text-xs">quantum</div>
                    <div className="p-2 bg-green-100 rounded text-center text-xs">physics</div>
                  </div>
                  
                  <div className="mt-4 text-center mb-2">Thinking Steps Assignment</div>
                  <div className="grid grid-cols-8 gap-1">
                    <div className="p-2 bg-gray-100 rounded text-center text-xs">1</div>
                    <div className="p-2 bg-gray-100 rounded text-center text-xs">1</div>
                    <div className="p-2 bg-gray-200 rounded text-center text-xs">2</div>
                    <div className="p-2 bg-gray-100 rounded text-center text-xs">1</div>
                    <div className="p-2 bg-gray-100 rounded text-center text-xs">1</div>
                    <div className="p-2 bg-gray-200 rounded text-center text-xs">2</div>
                    <div className="p-2 bg-gray-400 rounded text-center text-xs">4</div>
                    <div className="p-2 bg-gray-400 rounded text-center text-xs">4</div>
                  </div>
                  
                  <div className="mt-6 mb-2 text-center">Computational Intensity</div>
                  <div className="h-8 w-full bg-gray-200 rounded overflow-hidden flex">
                    <div className="bg-blue-300 h-full" style={{ width: '50%' }}>
                      <div className="text-xs p-1">Baseline</div>
                    </div>
                    <div className="bg-blue-500 h-full" style={{ width: '25%' }}>
                      <div className="text-xs text-white p-1">Extra Compute</div>
                    </div>
                  </div>
                  <div className="text-center text-sm mt-2">
                    Only 25% more computation for complex tokens
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
      
      {/* Residual Thinking Tab */}
      {activeTab === 'residual-thinking' && (
        <Card>
          <CardHeader>
            <CardTitle>Residual Thinking Connections</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div>
                <h3 className="font-bold text-lg mb-4">Concept</h3>
                <p className="mb-4">
                  Residual Thinking Connections allow the model to iteratively refine token 
                  representations through multiple thinking steps. Each thinking step builds
                  upon previous ones, focusing on different aspects of reasoning.
                </p>
                
                <div className="bg-green-50 p-4 rounded mb-4">
                  <h4 className="font-bold mb-2">Key Mechanism</h4>
                  <p>
                    A gating mechanism controls how much new information from each thinking step
                    is incorporated into the token representation. Later thinking steps have
                    more influence on complex tokens.
                  </p>
                </div>
                
                <div className="bg-blue-50 p-4 rounded">
                  <h4 className="font-bold mb-2">Implementation Pseudocode</h4>
                  <pre className="text-xs overflow-x-auto">
{`# Thinking block with residual connections
def thinking_block(x, step, max_steps):
    # Step bias increases influence of later steps
    step_bias = step / max_steps
    
    # Self-attention for reasoning
    attn_output = self_attention(x)
    
    # Gating mechanism
    gate = sigmoid(linear_layer([x, attn_output]))
    
    # Controlled residual connection
    x = x + dropout(attn_output) * gate * step_bias
    
    # Feed-forward with gated residual
    ffn_output = feed_forward(x)
    x = x + dropout(ffn_output) * gate * step_bias
    
    return x`}
                  </pre>
                </div>
              </div>
              
              <div>
                <h3 className="font-bold text-lg mb-4 text-center">Thinking Process</h3>
                <div className="bg-gray-50 rounded p-4">
                  <div className="flex flex-col items-center">
                    <div className="bg-blue-100 p-3 rounded w-64 text-center">
                      Initial Token Representation
                    </div>
                    <div className="h-8 w-1 bg-gray-300"></div>
                    
                    <div className="bg-purple-100 p-3 rounded w-64 text-center">
                      Thinking Step 1
                      <div className="text-xs mt-1">Basic Understanding</div>
                    </div>
                    <div className="h-8 w-1 bg-gray-300"></div>
                    
                    <div className="bg-purple-200 p-3 rounded w-64 text-center">
                      Thinking Step 2
                      <div className="text-xs mt-1">Contextual Integration</div>
                    </div>
                    <div className="h-8 w-1 bg-gray-300"></div>
                    
                    <div className="bg-purple-300 p-3 rounded w-64 text-center">
                      Thinking Step 3
                      <div className="text-xs mt-1">Deeper Relationships</div>
                    </div>
                    <div className="h-8 w-1 bg-gray-300"></div>
                    
                    <div className="bg-purple-400 p-3 rounded w-64 text-center">
                      Thinking Step 4
                      <div className="text-xs mt-1">Complex Reasoning</div>
                    </div>
                    <div className="h-8 w-1 bg-gray-300"></div>
                    
                    <div className="bg-green-100 p-3 rounded w-64 text-center">
                      Refined Token Representation
                    </div>
                  </div>
                  
                  <div className="mt-6 text-center">
                    <div className="font-bold mb-2">Gating Influence by Step</div>
                    <div className="flex justify-between items-center">
                      <div className="text-xs">Step 1</div>
                      <div className="h-4 w-16 bg-blue-200 rounded-sm"></div>
                      <div className="text-xs">25%</div>
                    </div>
                    <div className="flex justify-between items-center mt-1">
                      <div className="text-xs">Step 2</div>
                      <div className="h-4 w-24 bg-blue-300 rounded-sm"></div>
                      <div className="text-xs">50%</div>
                    </div>
                    <div className="flex justify-between items-center mt-1">
                      <div className="text-xs">Step 3</div>
                      <div className="h-4 w-32 bg-blue-400 rounded-sm"></div>
                      <div className="text-xs">75%</div>
                    </div>
                    <div className="flex justify-between items-center mt-1">
                      <div className="text-xs">Step 4</div>
                      <div className="h-4 w-40 bg-blue-500 rounded-sm"></div>
                      <div className="text-xs">100%</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
      
      {/* Comparison Tab */}
      {activeTab === 'comparison' && (
        <Card>
          <CardHeader>
            <CardTitle>ITT vs Standard Transformer</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div>
                <h3 className="font-bold text-lg mb-4">Standard Transformer</h3>
                <div className="p-4 bg-gray-50 rounded mb-4">
                  <h4 className="font-bold mb-2">Architecture</h4>
                  <ul className="list-disc ml-6 space-y-1">
                    <li>Fixed computation for all tokens</li>
                    <li>Same number of layers for all tokens</li>
                    <li>Simple residual connections</li>
                    <li>Uniform attention weight distribution</li>
                  </ul>
                </div>
                
                <div className="p-4 bg-red-50 rounded">
                  <h4 className="font-bold mb-2">Limitations</h4>
                  <ul className="list-disc ml-6 space-y-1">
                    <li>Wastes computation on simple tokens</li>
                    <li>Requires more parameters for complex reasoning</li>
                    <li>All tokens get same computational budget</li>
                    <li>Must grow model size for better reasoning</li>
                  </ul>
                </div>
                
                <div className="mt-4 font-bold text-center">
                  466M Parameters
                </div>
              </div>
              
              <div>
                <h3 className="font-bold text-lg mb-4">Inner Thinking Transformer</h3>
                <div className="p-4 bg-gray-50 rounded mb-4">
                  <h4 className="font-bold mb-2">Architecture</h4>
                  <ul className="list-disc ml-6 space-y-1">
                    <li>Dynamic computation allocation</li>
                    <li>Adaptive thinking depth per token</li>
                    <li>Gated residual thinking connections</li>
                    <li>Complexity-based processing</li>
                  </ul>
                </div>
                
                <div className="p-4 bg-green-50 rounded">
                  <h4 className="font-bold mb-2">Advantages</h4>
                  <ul className="list-disc ml-6 space-y-1">
                    <li>Focuses computation where needed</li>
                    <li>Better reasoning with fewer parameters</li>
                    <li>More efficient token processing</li>
                    <li>Improved performance on complex tasks</li>
                  </ul>
                </div>
                
                <div className="mt-4 font-bold text-center">
                  162M Parameters (65% reduction)
                </div>
              </div>
            </div>
            
            <div className="mt-8">
              <h3 className="font-bold text-lg mb-4 text-center">Performance Comparison</h3>
              <div className="grid grid-cols-3 gap-4 max-w-2xl mx-auto">
                <div className="p-4 bg-blue-50 rounded text-center">
                  <div className="text-sm font-medium mb-1">Reasoning Tasks</div>
                  <div className="grid grid-cols-2 gap-1">
                    <div className="p-1 bg-gray-100 rounded text-xs">Standard: 65%</div>
                    <div className="p-1 bg-green-100 rounded text-xs">ITT: 72%</div>
                  </div>
                </div>
                <div className="p-4 bg-blue-50 rounded text-center">
                  <div className="text-sm font-medium mb-1">Language Tasks</div>
                  <div className="grid grid-cols-2 gap-1">
                    <div className="p-1 bg-gray-100 rounded text-xs">Standard: 78%</div>
                    <div className="p-1 bg-green-100 rounded text-xs">ITT: 79%</div>
                  </div>
                </div>
                <div className="p-4 bg-blue-50 rounded text-center">
                  <div className="text-sm font-medium mb-1">Training Cost</div>
                  <div className="grid grid-cols-2 gap-1">
                    <div className="p-1 bg-gray-100 rounded text-xs">Standard: 100%</div>
                    <div className="p-1 bg-green-100 rounded text-xs">ITT: 40%</div>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default InnerThinkingTransformerGuide;
