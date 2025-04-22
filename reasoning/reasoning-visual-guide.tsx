import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';

const ReasoningGuide = () => {
  const [selectedTechnique, setSelectedTechnique] = useState('CoT');
  
  const techniques = {
    CoT: {
      name: "Chain of Thought (CoT)",
      description: "Encourages step-by-step reasoning before reaching a conclusion",
      diagram: (
        <div className="flex flex-col items-center">
          <div className="bg-blue-100 p-4 rounded mb-2">Question</div>
          <div className="text-2xl">↓</div>
          <div className="bg-green-100 p-4 rounded mb-2">Step 1: Understand</div>
          <div className="text-2xl">↓</div>
          <div className="bg-green-100 p-4 rounded mb-2">Step 2: Process</div>
          <div className="text-2xl">↓</div>
          <div className="bg-green-100 p-4 rounded mb-2">Step 3: Calculate</div>
          <div className="text-2xl">↓</div>
          <div className="bg-yellow-100 p-4 rounded">Final Answer</div>
        </div>
      ),
      example: `Q: If a train travels 60 miles in 1 hour, how far in 3 hours?

Let's think step by step:
1. Speed = 60 miles/hour
2. Time = 3 hours
3. Distance = Speed × Time
4. Distance = 60 × 3 = 180 miles

Answer: 180 miles`,
      advantages: [
        "Simple to implement",
        "Improves accuracy on complex tasks",
        "Works with zero-shot prompting"
      ],
      limitations: [
        "Can be verbose",
        "Errors propagate through steps",
        "Linear thinking only"
      ]
    },
    ToT: {
      name: "Tree of Thoughts (ToT)",
      description: "Explores multiple reasoning paths and evaluates their promise",
      diagram: (
        <div className="flex flex-col items-center">
          <div className="bg-blue-100 p-4 rounded mb-4">Question</div>
          <div className="flex justify-center space-x-8">
            <div className="flex flex-col items-center">
              <div className="text-xl">↙</div>
              <div className="bg-green-100 p-2 rounded mb-2">Thought 1</div>
              <div className="flex space-x-4">
                <div className="flex flex-col items-center">
                  <div className="text-lg">↙</div>
                  <div className="bg-green-200 p-2 rounded text-sm">1.1 ✓</div>
                </div>
                <div className="flex flex-col items-center">
                  <div className="text-lg">↘</div>
                  <div className="bg-red-200 p-2 rounded text-sm">1.2 ✗</div>
                </div>
              </div>
            </div>
            <div className="flex flex-col items-center">
              <div className="text-xl">↘</div>
              <div className="bg-green-100 p-2 rounded mb-2">Thought 2</div>
              <div className="flex space-x-4">
                <div className="flex flex-col items-center">
                  <div className="text-lg">↙</div>
                  <div className="bg-green-200 p-2 rounded text-sm">2.1 ✓</div>
                </div>
                <div className="flex flex-col items-center">
                  <div className="text-lg">↘</div>
                  <div className="bg-green-200 p-2 rounded text-sm">2.2 ✓</div>
                </div>
              </div>
            </div>
          </div>
          <div className="mt-4 bg-yellow-100 p-4 rounded">Best Path to Answer</div>
        </div>
      ),
      example: `Q: Game of 24: Make 24 using 4, 5, 6, 10

Branch 1: 4 + 5 + 6 + 10 = 25 ✗
Branch 2: (10 - 4) × (6 - 5) × 4 = 24 ✓
Branch 3: 10 × 6 - 4 × 5 = 40 ✗

Best path: Branch 2`,
      advantages: [
        "Explores multiple solutions",
        "Self-evaluates paths",
        "Allows backtracking"
      ],
      limitations: [
        "Computationally expensive",
        "Requires careful evaluation",
        "Complex to implement"
      ]
    },
    GoT: {
      name: "Graph of Thoughts (GoT)",
      description: "Creates complex reasoning graphs with merging and feedback",
      diagram: (
        <div className="flex flex-col items-center">
          <div className="bg-blue-100 p-4 rounded mb-4">Question</div>
          <div className="flex justify-center space-x-12">
            <div className="flex flex-col items-center">
              <div className="bg-green-100 p-2 rounded mb-2">Idea 1</div>
              <div className="text-xl">↘</div>
            </div>
            <div className="flex flex-col items-center">
              <div className="bg-green-100 p-2 rounded mb-2">Idea 2</div>
              <div className="text-xl">↙</div>
            </div>
          </div>
          <div className="bg-purple-100 p-4 rounded mb-4">Merged Idea</div>
          <div className="flex justify-center space-x-12">
            <div className="flex flex-col items-center">
              <div className="text-xl">↓</div>
              <div className="bg-orange-100 p-2 rounded">Critique</div>
            </div>
            <div className="flex flex-col items-center">
              <div className="text-xl">↓</div>
              <div className="bg-blue-200 p-2 rounded">Refinement</div>
            </div>
          </div>
          <div className="mt-4 bg-yellow-100 p-4 rounded">Final Answer</div>
        </div>
      ),
      example: `Q: Design a sustainable city

Idea 1: Green energy focus
Idea 2: Public transport priority
Merged: Integrated eco-transport
Critique: Needs residential planning
Refinement: Complete sustainable urban design`,
      advantages: [
        "Supports complex reasoning",
        "Allows idea merging",
        "Enables feedback loops"
      ],
      limitations: [
        "Most complex to implement",
        "High computational cost",
        "May over-complicate simple problems"
      ]
    },
    DoT: {
      name: "Diagram of Thought (DoT)",
      description: "Uses role-specific phases within a DAG structure",
      diagram: (
        <div className="flex flex-col items-center">
          <div className="bg-blue-100 p-4 rounded mb-2">Question</div>
          <div className="text-2xl">↓</div>
          <div className="bg-green-100 p-4 rounded mb-2">PROPOSER: Initial idea</div>
          <div className="text-2xl">↓</div>
          <div className="bg-red-100 p-4 rounded mb-2">CRITIC: Evaluation</div>
          <div className="text-2xl">↓</div>
          <div className="bg-yellow-100 p-4 rounded mb-2">REFINER: Improvement</div>
          <div className="text-2xl">↓</div>
          <div className="bg-purple-100 p-4 rounded mb-2">VERIFIER: Validation</div>
          <div className="text-2xl">↓</div>
          <div className="bg-green-200 p-4 rounded">Verified Answer</div>
        </div>
      ),
      example: `Q: Solve x² - 5x + 6 = 0

PROPOSER: Factor the quadratic
CRITIC: Need to find factors of 6 that sum to -5
REFINER: (x - 2)(x - 3) = 0
VERIFIER: x = 2 or x = 3 ✓`,
      advantages: [
        "Clear role separation",
        "Self-contained process",
        "Systematic verification"
      ],
      limitations: [
        "Fixed role sequence",
        "May be overkill for simple tasks",
        "Less flexible than GoT"
      ]
    }
  };
  
  return (
    <div className="w-full max-w-6xl mx-auto p-4">
      <h1 className="text-3xl font-bold mb-6">Structured Reasoning Techniques for LLMs</h1>
      
      {/* Technique Selector */}
      <div className="flex space-x-4 mb-8">
        {Object.keys(techniques).map((key) => (
          <button
            key={key}
            onClick={() => setSelectedTechnique(key)}
            className={`px-4 py-2 rounded ${
              selectedTechnique === key 
                ? 'bg-blue-500 text-white' 
                : 'bg-gray-200 hover:bg-gray-300'
            }`}
          >
            {key}
          </button>
        ))}
      </div>
      
      {/* Main Content */}
      <Card className="mb-8">
        <CardHeader>
          <CardTitle>{techniques[selectedTechnique].name}</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="mb-6">{techniques[selectedTechnique].description}</p>
          
          {/* Diagram */}
          <div className="mb-8 p-4 bg-gray-50 rounded">
            {techniques[selectedTechnique].diagram}
          </div>
          
          {/* Example */}
          <div className="mb-8">
            <h3 className="font-bold mb-2">Example:</h3>
            <pre className="bg-gray-100 p-4 rounded overflow-x-auto whitespace-pre-wrap">
              {techniques[selectedTechnique].example}
            </pre>
          </div>
          
          {/* Advantages and Limitations */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-bold mb-2">Advantages:</h3>
              <ul className="list-disc ml-6">
                {techniques[selectedTechnique].advantages.map((adv, i) => (
                  <li key={i}>{adv}</li>
                ))}
              </ul>
            </div>
            <div>
              <h3 className="font-bold mb-2">Limitations:</h3>
              <ul className="list-disc ml-6">
                {techniques[selectedTechnique].limitations.map((lim, i) => (
                  <li key={i}>{lim}</li>
                ))}
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
      
      {/* Comparison Table */}
      <Card>
        <CardHeader>
          <CardTitle>Technique Comparison</CardTitle>
        </CardHeader>
        <CardContent>
          <table className="w-full">
            <thead>
              <tr className="bg-gray-100">
                <th className="p-2 text-left">Technique</th>
                <th className="p-2 text-left">Complexity</th>
                <th className="p-2 text-left">Best For</th>
                <th className="p-2 text-left">Typical Improvement</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="p-2">CoT</td>
                <td className="p-2">Low</td>
                <td className="p-2">Math problems, logical reasoning</td>
                <td className="p-2">10-30%</td>
              </tr>
              <tr className="bg-gray-50">
                <td className="p-2">ToT</td>
                <td className="p-2">Medium</td>
                <td className="p-2">Planning, puzzles, creative tasks</td>
                <td className="p-2">50-70%</td>
              </tr>
              <tr>
                <td className="p-2">GoT</td>
                <td className="p-2">High</td>
                <td className="p-2">Complex problems, creativity</td>
                <td className="p-2">Variable</td>
              </tr>
              <tr className="bg-gray-50">
                <td className="p-2">DoT</td>
                <td className="p-2">Medium</td>
                <td className="p-2">Iterative refinement, verification</td>
                <td className="p-2">20-40%</td>
              </tr>
            </tbody>
          </table>
        </CardContent>
      </Card>
    </div>
  );
};

export default ReasoningGuide;
