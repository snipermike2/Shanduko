import React, { useState, useEffect } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, PieChart, Pie, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts';

const MetricsDashboard = () => {
  // State for metrics data
  const [metricsData, setMetricsData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    // Simulate API call - replace with actual API call when backend is ready
    const fetchMetricsData = async () => {
      try {
        // Simulating fetch delay
        setTimeout(() => {
          // Sample data - replace with actual API call
          setMetricsData(SAMPLE_DATA);
          setLoading(false);
        }, 1000);
      } catch (err) {
        setError("Failed to fetch metrics data");
        setLoading(false);
      }
    };

    fetchMetricsData();
  }, []);

  // Colors for charts
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d'];
  
  // Sample data structure - replace with actual API response
  const SAMPLE_DATA = {
    basic_metrics: {
      temperature: { rmse: 0.45, r2: 0.92, mape: 3.5 },
      ph: { rmse: 0.15, r2: 0.94, mape: 2.1 },
      dissolved_oxygen: { rmse: 0.35, r2: 0.89, mape: 4.2 },
      turbidity: { rmse: 0.55, r2: 0.87, mape: 5.1 }
    },
    critical_metrics: {
      temperature: { sensitivity: 0.92, specificity: 0.94, precision: 0.89, critical_f1: 0.90 },
      ph: { sensitivity: 0.88, specificity: 0.96, precision: 0.92, critical_f1: 0.90 },
      dissolved_oxygen: { sensitivity: 0.85, specificity: 0.93, precision: 0.88, critical_f1: 0.86 },
      turbidity: { sensitivity: 0.82, specificity: 0.91, precision: 0.85, critical_f1: 0.83 }
    },
    water_quality_standards: {
      drinking_water: {
        compliance_rate: 0.95,
        parameters: {
          temperature: 0.96,
          ph: 0.98,
          dissolved_oxygen: 0.94,
          turbidity: 0.92
        }
      },
      aquatic_life: {
        compliance_rate: 0.92,
        parameters: {
          temperature: 0.93,
          ph: 0.95,
          dissolved_oxygen: 0.91,
          turbidity: 0.89
        }
      },
      irrigation: {
        compliance_rate: 0.97,
        parameters: {
          temperature: 0.98,
          ph: 0.97,
          dissolved_oxygen: 0.96,
          turbidity: 0.95
        }
      }
    },
    ecological_impact: {
      temperature: {
        optimal_time_ratio: 0.75,
        warning_time_ratio: 0.20,
        critical_time_ratio: 0.05
      },
      ph: {
        optimal_time_ratio: 0.82,
        warning_time_ratio: 0.15,
        critical_time_ratio: 0.03
      },
      dissolved_oxygen: {
        optimal_time_ratio: 0.70,
        warning_time_ratio: 0.25,
        critical_time_ratio: 0.05
      },
      turbidity: {
        optimal_time_ratio: 0.65,
        warning_time_ratio: 0.30,
        critical_time_ratio: 0.05
      }
    },
    treatment_metrics: {
      temperature: {
        treatment_intensity: 0.25,
        treatment_frequency: 0.15
      },
      ph: {
        treatment_intensity: 0.18,
        treatment_frequency: 0.12
      },
      dissolved_oxygen: {
        treatment_intensity: 0.30,
        treatment_frequency: 0.20
      },
      turbidity: {
        treatment_intensity: 0.35,
        treatment_frequency: 0.25
      }
    },
    temporal_metrics: {
      "1h": {
        temperature: { rmse: 0.22 },
        ph: { rmse: 0.08 },
        dissolved_oxygen: { rmse: 0.18 },
        turbidity: { rmse: 0.25 }
      },
      "6h": {
        temperature: { rmse: 0.35 },
        ph: { rmse: 0.12 },
        dissolved_oxygen: { rmse: 0.28 },
        turbidity: { rmse: 0.38 }
      },
      "24h": {
        temperature: { rmse: 0.48 },
        ph: { rmse: 0.18 },
        dissolved_oxygen: { rmse: 0.42 },
        turbidity: { rmse: 0.52 }
      }
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <div className="text-lg font-medium mb-2">Loading metrics data...</div>
          <div className="w-16 h-16 border-4 border-t-blue-500 border-blue-200 rounded-full animate-spin mx-auto"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 bg-red-50 border border-red-200 rounded-lg">
        <h2 className="text-xl font-bold text-red-700 mb-2">Error Loading Data</h2>
        <p className="text-red-600">{error}</p>
        <button 
          className="mt-4 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
          onClick={() => window.location.reload()}
        >
          Retry
        </button>
      </div>
    );
  }

  const data = metricsData || SAMPLE_DATA;

  // Navigation tabs
  const renderTabs = () => (
    <div className="flex border-b border-gray-200 mb-6">
      <button
        className={`px-4 py-2 font-medium ${activeTab === 'overview' ? 'text-blue-600 border-b-2 border-blue-600' : 'text-gray-500 hover:text-blue-500'}`}
        onClick={() => setActiveTab('overview')}
      >
        Overview
      </button>
      <button
        className={`px-4 py-2 font-medium ${activeTab === 'compliance' ? 'text-blue-600 border-b-2 border-blue-600' : 'text-gray-500 hover:text-blue-500'}`}
        onClick={() => setActiveTab('compliance')}
      >
        Compliance
      </button>
      <button
        className={`px-4 py-2 font-medium ${activeTab === 'ecological' ? 'text-blue-600 border-b-2 border-blue-600' : 'text-gray-500 hover:text-blue-500'}`}
        onClick={() => setActiveTab('ecological')}
      >
        Ecological Impact
      </button>
      <button
        className={`px-4 py-2 font-medium ${activeTab === 'prediction' ? 'text-blue-600 border-b-2 border-blue-600' : 'text-gray-500 hover:text-blue-500'}`}
        onClick={() => setActiveTab('prediction')}
      >
        Prediction Accuracy
      </button>
    </div>
  );

  // Basic Metrics Chart Component
  const BasicMetricsChart = () => (
    <div className="bg-white p-6 rounded-lg shadow-md mb-6">
      <h2 className="text-xl font-bold mb-4">Model Performance Metrics</h2>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={Object.entries(data.basic_metrics).map(([param, metrics]) => ({
              parameter: param.charAt(0).toUpperCase() + param.slice(1),
              RMSE: metrics.rmse,
              'R²': metrics.r2,
              MAPE: metrics.mape / 100
            }))}
            margin={{ top: 20, right: 30, left: 20, bottom: 70 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="parameter" angle={-45} textAnchor="end" height={70} />
            <YAxis />
            <Tooltip formatter={(value, name) => [name === 'MAPE' ? `${(value * 100).toFixed(2)}%` : value.toFixed(3), name]} />
            <Legend />
            <Bar dataKey="RMSE" fill="#8884d8" />
            <Bar dataKey="R²" fill="#82ca9d" />
            <Bar dataKey="MAPE" fill="#ffc658" />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-4 text-sm text-gray-600">
        <p><strong>RMSE</strong> (Root Mean Square Error): Lower values indicate better prediction accuracy.</p>
        <p><strong>R²</strong> (R-squared): Values closer to 1 indicate better fit of the model.</p>
        <p><strong>MAPE</strong> (Mean Absolute Percentage Error): Lower values indicate more accurate predictions.</p>
      </div>
    </div>
  );

  // Critical Detection Metrics Chart
  const CriticalMetricsChart = () => (
    <div className="bg-white p-6 rounded-lg shadow-md mb-6">
      <h2 className="text-xl font-bold mb-4">Critical Event Detection</h2>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart outerRadius={90} width={730} height={250} data={
            Object.entries(data.critical_metrics).map(([param, metrics]) => ({
              parameter: param.charAt(0).toUpperCase() + param.slice(1),
              Sensitivity: metrics.sensitivity,
              Specificity: metrics.specificity,
              Precision: metrics.precision,
              F1: metrics.critical_f1
            }))
          }>
            <PolarGrid />
            <PolarAngleAxis dataKey="parameter" />
            <PolarRadiusAxis angle={30} domain={[0, 1]} />
            <Radar name="Sensitivity" dataKey="Sensitivity" stroke="#8884d8" fill="#8884d8" fillOpacity={0.6} />
            <Radar name="Specificity" dataKey="Specificity" stroke="#82ca9d" fill="#82ca9d" fillOpacity={0.6} />
            <Radar name="Precision" dataKey="Precision" stroke="#ffc658" fill="#ffc658" fillOpacity={0.6} />
            <Radar name="F1" dataKey="F1" stroke="#ff8042" fill="#ff8042" fillOpacity={0.6} />
            <Legend />
            <Tooltip formatter={(value) => [(value * 100).toFixed(1) + '%']} />
          </RadarChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-4 text-sm text-gray-600">
        <p><strong>Sensitivity</strong>: Ability to correctly detect critical events when they occur.</p>
        <p><strong>Specificity</strong>: Ability to correctly identify normal conditions.</p>
        <p><strong>Precision</strong>: Percentage of detected critical events that are actually critical.</p>
        <p><strong>F1 Score</strong>: Overall balance between sensitivity and precision.</p>
      </div>
    </div>
  );

  // Compliance Chart Component
  const ComplianceChart = () => (
    <div className="bg-white p-6 rounded-lg shadow-md mb-6">
      <h2 className="text-xl font-bold mb-4">Water Quality Standards Compliance</h2>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={Object.entries(data.water_quality_standards).map(([use, data]) => ({
              use: use.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' '),
              'Compliance Rate': data.compliance_rate * 100
            }))}
            margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="use" />
            <YAxis domain={[0, 100]} label={{ value: 'Compliance (%)', angle: -90, position: 'insideLeft' }} />
            <Tooltip formatter={(value) => [`${value.toFixed(1)}%`, 'Compliance Rate']} />
            <Legend />
            <Bar dataKey="Compliance Rate" fill="#82ca9d" />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-4 text-sm text-gray-600">
        <p>This chart shows the percentage of predicted water quality measurements that comply with standards for different water uses.</p>
        <p>Higher compliance rates indicate the water is more suitable for the specified use category.</p>
      </div>
    </div>
  );

  // Parameter Compliance Chart
  const ParameterComplianceChart = () => (
    <div className="bg-white p-6 rounded-lg shadow-md mb-6">
      <h2 className="text-xl font-bold mb-4">Parameter-Specific Compliance</h2>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={
              Object.keys(data.water_quality_standards.drinking_water.parameters).map(param => {
                return {
                  parameter: param.charAt(0).toUpperCase() + param.slice(1),
                  'Drinking Water': data.water_quality_standards.drinking_water.parameters[param] * 100,
                  'Aquatic Life': data.water_quality_standards.aquatic_life.parameters[param] * 100,
                  'Irrigation': data.water_quality_standards.irrigation.parameters[param] * 100
                };
              })
            }
            margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="parameter" />
            <YAxis domain={[0, 100]} label={{ value: 'Compliance (%)', angle: -90, position: 'insideLeft' }} />
            <Tooltip formatter={(value) => [`${value.toFixed(1)}%`, 'Compliance']} />
            <Legend />
            <Bar dataKey="Drinking Water" fill="#0088FE" />
            <Bar dataKey="Aquatic Life" fill="#00C49F" />
            <Bar dataKey="Irrigation" fill="#FFBB28" />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-4 text-sm text-gray-600">
        <p>This chart breaks down compliance by parameter across different water use categories.</p>
        <p>It helps identify which parameters are most challenging to maintain within standards.</p>
      </div>
    </div>
  );

  // Ecological Impact Chart
  const EcologicalImpactChart = () => (
    <div className="bg-white p-6 rounded-lg shadow-md mb-6">
      <h2 className="text-xl font-bold mb-4">Ecological Impact Distribution</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {Object.entries(data.ecological_impact).map(([param, metrics]) => (
          <div key={param} className="h-64">
            <h3 className="text-center font-medium mb-2">{param.charAt(0).toUpperCase() + param.slice(1)}</h3>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={[
                    { name: 'Optimal', value: metrics.optimal_time_ratio },
                    { name: 'Warning', value: metrics.warning_time_ratio },
                    { name: 'Critical', value: metrics.critical_time_ratio }
                  ]}
                  cx="50%"
                  cy="50%"
                  labelLine={true}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                >
                  {[
                    { name: 'Optimal', value: metrics.optimal_time_ratio },
                    { name: 'Warning', value: metrics.warning_time_ratio },
                    { name: 'Critical', value: metrics.critical_time_ratio }
                  ].map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => [(value * 100).toFixed(1) + '%']} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        ))}
      </div>
      <div className="mt-4 text-sm text-gray-600">
        <p><strong>Optimal</strong>: Percentage of time water quality is within ideal range for aquatic life.</p>
        <p><strong>Warning</strong>: Percentage of time water quality may cause mild stress to aquatic life.</p>
        <p><strong>Critical</strong>: Percentage of time water quality may be harmful to aquatic life.</p>
      </div>
    </div>
  );

  // Treatment Requirements Chart
  const TreatmentRequirementsChart = () => (
    <div className="bg-white p-6 rounded-lg shadow-md mb-6">
      <h2 className="text-xl font-bold mb-4">Treatment Requirements</h2>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={Object.entries(data.treatment_metrics).map(([param, metrics]) => ({
              parameter: param.charAt(0).toUpperCase() + param.slice(1),
              'Treatment Intensity': metrics.treatment_intensity * 100,
              'Treatment Frequency': metrics.treatment_frequency * 100
            }))}
            margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="parameter" />
            <YAxis domain={[0, 100]} label={{ value: 'Percentage (%)', angle: -90, position: 'insideLeft' }} />
            <Tooltip formatter={(value) => [`${value.toFixed(1)}%`]} />
            <Legend />
            <Bar dataKey="Treatment Intensity" fill="#8884d8" />
            <Bar dataKey="Treatment Frequency" fill="#82ca9d" />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-4 text-sm text-gray-600">
        <p><strong>Treatment Intensity</strong>: Average level of treatment required to meet standards (higher values indicate more intensive treatment).</p>
        <p><strong>Treatment Frequency</strong>: Percentage of time treatment is required to meet water quality standards.</p>
      </div>
    </div>
  );

  // Prediction Accuracy Chart
  const PredictionAccuracyChart = () => (
    <div className="bg-white p-6 rounded-lg shadow-md mb-6">
      <h2 className="text-xl font-bold mb-4">Prediction Accuracy by Time Horizon</h2>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="horizon" 
              type="category" 
              allowDuplicatedCategory={false} 
              domain={['1h', '6h', '24h']}
            />
            <YAxis label={{ value: 'RMSE', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Legend />
            {Object.keys(data.temporal_metrics['1h']).map((param, index) => (
              <Line
                key={param}
                type="monotone"
                dataKey="rmse"
                name={param.charAt(0).toUpperCase() + param.slice(1)}
                stroke={COLORS[index % COLORS.length]}
                activeDot={{ r: 8 }}
                data={[
                  { horizon: '1h', rmse: data.temporal_metrics['1h'][param].rmse },
                  { horizon: '6h', rmse: data.temporal_metrics['6h'][param].rmse },
                  { horizon: '24h', rmse: data.temporal_metrics['24h'][param].rmse }
                ]}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-4 text-sm text-gray-600">
        <p>This chart shows how prediction accuracy (measured by RMSE) changes as the time horizon increases.</p>
        <p>Lower RMSE values indicate more accurate predictions. Typically, shorter time horizons yield more accurate predictions.</p>
      </div>
    </div>
  );

  // Render different content based on active tab
  const renderContent = () => {
    switch (activeTab) {
      case 'overview':
        return (
          <>
            <BasicMetricsChart />
            <CriticalMetricsChart />
          </>
        );
      case 'compliance':
        return (
          <>
            <ComplianceChart />
            <ParameterComplianceChart />
          </>
        );
      case 'ecological':
        return (
          <>
            <EcologicalImpactChart />
            <TreatmentRequirementsChart />
          </>
        );
      case 'prediction':
        return (
          <>
            <PredictionAccuracyChart />
          </>
        );
      default:
        return <BasicMetricsChart />;
    }
  };

  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-2 text-blue-800">Water Quality Metrics Dashboard</h1>
        <p className="text-gray-600 mb-6">
          A comprehensive analysis of water quality prediction metrics and compliance standards
        </p>
        
        {/* Navigation Tabs */}
        {renderTabs()}
        
        {/* Dashboard Content */}
        {renderContent()}
        
        {/* Footer */}
        <div className="mt-8 pt-4 border-t border-gray-200 text-center text-gray-500 text-sm">
          <p>Shanduko Water Quality Monitoring System</p>
          <p>Last updated: {new Date().toLocaleDateString()}</p>
        </div>
      </div>
    </div>
  );
};

export default MetricsDashboard;