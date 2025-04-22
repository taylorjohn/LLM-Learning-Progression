import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

const BitNetLearningGuide = () => {
  const [currentStep, setCurrentStep] = useState(0);
  
  const learningSteps = [
    {
      title: "Understanding Quantization Basics",
      content: "Learn how BitNet a4.8 reduces memory by using 1-bit weights and 4-bit activations instead of 32-bit floats.",
      code: `# Standard float32 weights
weights = torch.randn(64, 64)  # 32 bits per value

# BitNet 1-bit weights
bitnet_weights = torch.sign(weights)  # {-1, 1}

# Memory reduction: 32x for weights!`
    },
    {
      title: "Implementing 4-bit Activation Quantization",
      content: "Master the hybrid approach that combines sparsification and 4-bit quantization for activations.",
      code: `def quantize_4bit(x):
    scale = torch.abs(x).max() / 7.5
    x_quantized = torch.round(x / scale).clamp(-7, 8)
    
    # Sparsification
    sparse_mask = torch.abs(x) < 0.1
    x_quantized[sparse_mask] = 0
    
    return x_quantized * scale`
    },
    {
      title: "Converting Linear Layers",
      content: "Transform standard linear layers into BitLinear layers with 1-bit weights.",
      code: `class BitLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        # 1-bit quantization during forward
        w_1bit = torch.sign(self.linear.weight)
        return F.linear(x, w_1bit, self.linear.bias)`
    },
    {
      title: "Quantizing Attention Mechanisms",
      content: "Apply quantization to attention layers for efficient computation.",
      code: `# In attention mechanism
q, k, v = self.qkv_proj(x).chunk(3, dim=-1)

# Apply 4-bit quantization to Q and K
q = quantize_4bit(q)
k = quantize_4bit(k)

# Compute attention as usual
scores = torch.matmul(q, k.transpose(-1, -2))`
    },
    {
      title: "Integrating into Complete LLM",
      content: "Put it all together in a full transformer-based language model.",
      code: `class BitNetLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = BitNetTransformer(embed_dim)
        self.lm_head = BitLinear(embed_dim, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = quantize_4bit(x)  # Quantize embeddings
        x = self.transformer(x)
        return self.lm_head(x)`
    }
  ];
  
  const memoryComparisonData = [
    { component: 'Weights', standard: 32, bitnet: 1 },
    { component: 'Activations', standard: 32, bitnet: 4 },
    { component: 'Gradients', standard: 32, bitnet: 8 }
  ];
  
  const memoryDistribution = [
    { name: 'Weights', value: 60, color: '#8884d8' },
    { name: 'Activations', value: 30, color: '#82ca9d' },
    { name: 'Others', value: 10, color: '#ffc658' }
  ];
  
  return (
    <div className="w-full max-w-6xl mx-auto p-4">
      <h1 className="text-3xl font-bold mb-6">BitNet a4.8 Learning Guide</h1>
      <p className="text-lg mb-8">
        Learn how to integrate BitNet a4.8 quantization into your LLM projects step by step.
      </p>
      
      {/* Progress Indicator */}
      <div className="mb-8">
        <div className="flex justify-between mb-2">
          {learningSteps.map((_, index) => (
            <button
              key={index}
              onClick={() => setCurrentStep(index)}
              className={`w-10 h-10 rounded-full flex items-center justify-center ${
                index === currentStep 
                  ? 'bg-blue-500 text-white' 
                  : index < currentStep 
                    ? 'bg-green-500 text-white'
                    : 'bg-gray-200'
              }`}
            >
              {index + 1}
            </button>
          ))}
        </div>
        <div className="w-full bg-gray-200 h-2 rounded">
          <div 
            className="bg-blue-500 h-2 rounded transition-all duration-300"
            style={{ width: `${(currentStep + 1) / learningSteps.length * 100}%` }}
          />
        </div>
      </div>
      
      {/* Current Step Content */}
      <Card className="mb-8">
        <CardHeader>
          <CardTitle>Step {currentStep + 1}: {learningSteps[currentStep].title}</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="mb-4">{learningSteps[currentStep].content}</p>
          <pre className="bg-gray-100 p-4 rounded overflow-x-auto">
            <code>{learningSteps[currentStep].code}</code>
          </pre>
          <div className="flex justify-between mt-4">
            <button
              onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
              disabled={currentStep === 0}
              className={`px-4 py-2 rounded ${
                currentStep === 0 
                  ? 'bg-gray-300 cursor-not-allowed' 
                  : 'bg-blue-500 text-white hover:bg-blue-600'
              }`}
            >
              Previous
            </button>
            <button
              onClick={() => setCurrentStep(Math.min(learningSteps.length - 1, currentStep + 1))}
              disabled={currentStep === learningSteps.length - 1}
              className={`px-4 py-2 rounded ${
                currentStep === learningSteps.length - 1 
                  ? 'bg-gray-300 cursor-not-allowed' 
                  : 'bg-blue-500 text-white hover:bg-blue-600'
              }`}
            >
              Next
            </button>
          </div>
        </CardContent>
      </Card>
      
      {/* Memory Visualization */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <Card>
          <CardHeader>
            <CardTitle>Bits per Component Comparison</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={memoryComparisonData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="component" />
                  <YAxis label={{ value: 'Bits', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="standard" stroke="#ff7300" name="Standard (32-bit)" />
                  <Line type="monotone" dataKey="bitnet" stroke="#82ca9d" name="BitNet a4.8" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle>Memory Distribution in LLMs</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={memoryDistribution}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({name, percent}) => `${name} (${(percent * 100).toFixed(0)}%)`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {memoryDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>
      
      {/* Key Concepts Summary */}
      <Card>
        <CardHeader>
          <CardTitle>Key Concepts</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 bg-blue-50 rounded">
              <h3 className="font-bold mb-2">1-bit Weights</h3>
              <p>Reduce weight precision from 32 bits to just 1 bit ({'{-1, 1}'}). Achieves 32x memory reduction for weights.</p>
            </div>
            <div className="p-4 bg-green-50 rounded">
              <h3 className="font-bold mb-2">4-bit Activations</h3>
              <p>Quantize activations to 16 levels (-7 to 8). Provides 8x reduction while maintaining performance.</p>
            </div>
            <div className="p-4 bg-yellow-50 rounded">
              <h3 className="font-bold mb-2">Sparsification</h3>
              <p>Combine 1-bit masks for near-zero values with 4-bit quantization for better efficiency.</p>
            </div>
          </div>
        </CardContent>
      </Card>
      
      {/* Performance Impact */}
      <Card className="mt-8">
        <CardHeader>
          <CardTitle>Performance Impact</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="list-disc ml-6 space-y-2">
            <li><strong>Memory:</strong> Up to 8x reduction overall</li>
            <li><strong>Inference Speed:</strong> 2-3x faster on consumer hardware</li>
            <li><strong>Accuracy:</strong> Within 2-3% of full precision models</li>
            <li><strong>Training:</strong> Can use larger batch sizes and models</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  );
};

export default BitNetLearningGuide;
