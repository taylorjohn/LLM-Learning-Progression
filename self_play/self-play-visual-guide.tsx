import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const SelfPlayGuide = () => {
  const [activeStep, setActiveStep] = useState(0);
  
  const steps = [
    {
      title: "Generation Phase",
      description: "Model generates multiple responses for each prompt"
    },
    {
      title: "Evaluation Phase",
      description: "Responses are scored based on quality metrics"
    },
    {
      title: "Selection Phase",
      description: "High-quality responses are added to training dataset"
    },
    {
      title: "Fine-tuning Phase",
      description: "Model is fine-tuned on self-generated high-quality data"
    },
    {
      title: "Iteration",
      description: "Process repeats to continuously improve the model"
    }
  ];
  
  const performanceData = [
    { iteration: 0, accuracy: 40, quality: 35 },
    { iteration: 5, accuracy: 55, quality: 50 },
    { iteration: 10, accuracy: 65, quality: 62 },
    { iteration: 15, accuracy: 72, quality: 70 },
    { iteration: 20, accuracy: 78, quality: 76 },
    { iteration: 25, accuracy: 82, quality: 80 },
  ];
  
  return (
    <div className="w-full max-w-6xl mx-auto p-4">
      <h1 className="text-3xl font-bold mb-6">Self-Play Fine-Tuning for LLMs</h1>
      
      {/* Process Overview */}
      <Card className="mb-8">
        <CardHeader>
          <CardTitle>Self-Play Training Loop</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex justify-between items-center mb-8">
            {steps.map((step, index) => (
              <div 
                key={index}
                className={`flex flex-col items-center ${
                  index === activeStep ? 'text-blue-600' : 'text-gray-500'
                }`}
                onClick={() => setActiveStep(index)}
              >
                <div className={`w-12 h-12 rounded-full flex items-center justify-center mb-2 cursor-pointer ${
                  index === activeStep ? 'bg-blue-500 text-white' : 'bg-gray-200'
                }`}>
                  {index + 1}
                </div>
                <div className="text-sm font-medium text-center">{step.title}</div>
              </div>
            ))}
          </div>
          
          <div className="bg-gray-50 p-4 rounded">
            <h3 className="font-bold mb-2">{steps[activeStep].title}</h3>
            <p>{steps[activeStep].description}</p>
          </div>
        </CardContent>
      </Card>
      
      {/* Detailed Process */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <Card>
          <CardHeader>
            <CardTitle>Generation & Evaluation</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="p-4 bg-gray-50 rounded">
                <h4 className="font-bold mb-2">Prompt:</h4>
                <p className="text-blue-600">"What is the capital of France?"</p>
              </div>
              
              <div className="space-y-2">
                <div className="p-3 bg-green-50 rounded flex justify-between">
                  <span>"Paris is the capital of France."</span>
                  <span className="text-green-600 font-bold">Score: 0.9</span>
                </div>
                <div className="p-3 bg-yellow-50 rounded flex justify-between">
                  <span>"The capital is Paris."</span>
                  <span className="text-yellow-600 font-bold">Score: 0.7</span>
                </div>
                <div className="p-3 bg-red-50 rounded flex justify-between">
                  <span>"I'm not sure."</span>
                  <span className="text-red-600 font-bold">Score: 0.2</span>
                </div>
              </div>
              
              <p className="text-sm text-gray-600">
                Only responses with scores above 0.5 are selected for training
              </p>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle>Evaluation Criteria</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <h4 className="font-bold mb-2">QA Tasks:</h4>
                <ul className="list-disc ml-6">
                  <li>Answers the question directly</li>
                  <li>Appropriate response length</li>
                  <li>Contains relevant content</li>
                </ul>
              </div>
              
              <div>
                <h4 className="font-bold mb-2">Math Tasks:</h4>
                <ul className="list-disc ml-6">
                  <li>Contains numerical answer</li>
                  <li>Result is reasonable</li>
                  <li>Shows calculation steps</li>
                </ul>
              </div>
              
              <div>
                <h4 className="font-bold mb-2">Reasoning Tasks:</h4>
                <ul className="list-disc ml-6">
                  <li>Logical flow of ideas</li>
                  <li>Uses connective words</li>
                  <li>Provides explanation</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
      
      {/* Performance Improvement */}
      <Card className="mb-8">
        <CardHeader>
          <CardTitle>Performance Improvement Over Time</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="iteration" label={{ value: 'Iteration', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Score (%)', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="accuracy" stroke="#8884d8" name="Accuracy" />
                <Line type="monotone" dataKey="quality" stroke="#82ca9d" name="Response Quality" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
      
      {/* Key Benefits */}
      <Card>
        <CardHeader>
          <CardTitle>Benefits of Self-Play Fine-Tuning</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 bg-blue-50 rounded">
              <h3 className="font-bold mb-2">Continuous Improvement</h3>
              <p>Model iteratively improves by learning from its best outputs</p>
            </div>
            <div className="p-4 bg-green-50 rounded">
              <h3 className="font-bold mb-2">Domain Adaptation</h3>
              <p>Adapts to specific tasks without external labeled data</p>
            </div>
            <div className="p-4 bg-yellow-50 rounded">
              <h3 className="font-bold mb-2">Quality Control</h3>
              <p>Only high-quality responses are used for training</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default SelfPlayGuide;
