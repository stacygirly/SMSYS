/* import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Line, Pie } from 'react-chartjs-2';
import { Chart, LineElement, LinearScale, Title, Tooltip, Legend, TimeScale, PointElement, ArcElement } from 'chart.js';
import 'chartjs-adapter-date-fns';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';
import 'bootstrap/dist/css/bootstrap.min.css';
import './dashboard.css';
import StressManagementTips from './stress managemnet tips/smt';

Chart.register(LineElement, LinearScale, Title, Tooltip, Legend, TimeScale, PointElement, ArcElement);

const Dashboard = () => {
    const [allStressData, setAllStressData] = useState([]);
    const [filteredStressData, setFilteredStressData] = useState([]);
    const [selectedDate, setSelectedDate] = useState(new Date());

    useEffect(() => {
        axios.get('http://localhost:5000/stress_dashboard', { withCredentials: true })
            .then(response => {
                console.log('API response data:', response.data);
                setAllStressData(response.data);
                filterDataByDate(response.data, new Date());
            })
            .catch(error => {
                console.error('Error fetching stress data:', error);
            });
    }, []);

    const filterDataByDate = (data, date) => {
        const selectedDateStart = new Date(date.setHours(0, 0, 0, 0));
        const selectedDateEnd = new Date(date.setHours(23, 59, 59, 999));
        const filteredData = data.filter(item => {
            const itemDate = new Date(item.timestamp);
            return itemDate >= selectedDateStart && itemDate <= selectedDateEnd;
        });
        setFilteredStressData(filteredData);
    };

    useEffect(() => {
        filterDataByDate(allStressData, selectedDate);
    }, [selectedDate, allStressData]);

    const lineChartData = {
        labels: filteredStressData.map(data => new Date(data.timestamp)),
        datasets: [
            {
                label: 'Stress Level Over Time',
                data: filteredStressData.map(data => data.model_prediction),
                fill: false,
                borderColor: '#1e90ff',  // Updated line color to match theme
                tension: 0.3,  // Slight curve for a smoother line
                pointBackgroundColor: filteredStressData.map(data =>
                    data.emotion === 'stressed' ? '#dc3545' : '#28a745'  // Updated red and green colors
                ),
                pointBorderColor: filteredStressData.map(data =>
                    data.emotion === 'stressed' ? '#dc3545' : '#28a745'
                ),
                pointRadius: 5,
                pointHoverRadius: 8,
            },
        ],
    };

    const lineChartOptions = {
        scales: {
            x: {
                type: 'time',
                time: {
                    unit: 'hour',
                    displayFormats: {
                        hour: 'ha',
                    },
                    stepSize: 1,
                },
                title: {
                    display: true,
                    text: 'Time',
                    color: '#ffffff',  // Updated text color to white
                },
                ticks: {
                    color: '#ffffff',
                },
                grid: {
                    color: 'rgba(255,255,255,0.2)',
                },
            },
            y: {
                title: {
                    display: true,
                    text: 'Model Prediction Value',
                    color: '#ffffff',
                },
                ticks: {
                    color: '#ffffff',
                },
                grid: {
                    color: 'rgba(255,255,255,0.2)',
                },
            },
        },
        plugins: {
            legend: {
                display: true,
                labels: {
                    color: '#ffffff',
                },
            },
        },
        animation: {
            duration: 1500,  // Animation duration in milliseconds
            easing: 'easeInOutQuad',  // Easing function for a smooth animation
        }
    };

    const stressCounts = filteredStressData.reduce((acc, data) => {
        acc[data.emotion] = (acc[data.emotion] || 0) + 1;
        return acc;
    }, {});

    const totalEntries = filteredStressData.length;
    const pieChartData = {
        labels: ['Stressed', 'Not Stressed'],
        datasets: [
            {
                data: [
                    stressCounts.stressed || 0,
                    stressCounts['not stressed'] || 0
                ],
                backgroundColor: ['#dc3545', '#28a745'],  // Updated pie chart colors
                borderColor: ['#721c24', '#155724'],  // Darker borders to match the theme
                hoverOffset: 4,  // Slightly expand the slices on hover
            }
        ]
    };

    const pieChartOptions = {
        plugins: {
            legend: {
                display: true,
                labels: {
                    color: '#ffffff',
                },
            },
            tooltip: {
                callbacks: {
                    label: function (context) {
                        let label = context.label || '';
                        if (label) {
                            label += ': ';
                        }
                        label += Math.round(context.raw * 100) / 100;
                        return label;
                    }
                }
            },
        },
        animation: {
            animateScale: true,  // Animate the scaling of the chart
            animateRotate: true,  // Animate the rotation of the chart
        }
    };

    return (
        <>
            <div className="dashboard-container">
                <div className="dashboard-content">
                    <div className="charts-section">
                        <div className="controls">
                            <label style={{ marginTop: '5px', fontWeight: 'bold', color: 'white' }}>Select Date: </label>
                            <DatePicker
                                selected={selectedDate}
                                onChange={(date) => setSelectedDate(date)}
                                className="custom-datepicker form-control"
                            />
                        </div>
                        <div className="row chart-wrapper">
                            <div className="chart col-md-7">
                                <h3 className="chart-title">Stress Level Over Time</h3>
                                {filteredStressData.length > 0 ? (
                                    <Line data={lineChartData} options={lineChartOptions} />
                                ) : (
                                    <p style={{ color: 'white' }}>No stress indicators found.</p>
                                )}
                            </div>
                            <div className="pie-chart-container col-md-5">
                                <h3 className="chart-title">Stress Summary</h3>
                                {totalEntries > 0 ? (
                                    <Pie data={pieChartData} options={pieChartOptions} />
                                ) : (
                                    <p style={{ color: 'white' }}>No data for pie chart.</p>
                                )}
                                <div className="pie-chart-info">
                                    <h3 className="chart-title">Stress Data Summary</h3>
                                    <ul style={{ color: 'white' }}>
                                        <li>Total Entries: {totalEntries}</li>
                                        <li>Stressed: {stressCounts.stressed || 0}</li>
                                        <li>Not Stressed: {stressCounts['not stressed'] || 0}</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                    <StressManagementTips />
                </div>
            </div>
        </>
    );
};

export default Dashboard;
 */
import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Line, Pie } from 'react-chartjs-2';
import { Chart, LineElement, LinearScale, Title, Tooltip, Legend, TimeScale, PointElement, ArcElement } from 'chart.js';
import 'chartjs-adapter-date-fns';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';
import 'bootstrap/dist/css/bootstrap.min.css';
import './dashboard.css';
import StressManagementTips from './stress managemnet tips/smt';

Chart.register(LineElement, LinearScale, Title, Tooltip, Legend, TimeScale, PointElement, ArcElement);

// ✅ Parse ISO strings or {$date: "..."} from Mongo
const toDate = (t) => new Date(typeof t === 'string' ? t : t?.$date || t || 0);

const Dashboard = () => {
  const [allStressData, setAllStressData] = useState([]);
  const [filteredStressData, setFilteredStressData] = useState([]);
  const [selectedDate, setSelectedDate] = useState(new Date());

  useEffect(() => {
    axios
      .get('https://persuasive.research.cs.dal.ca/smsys/stress_dashboard', { withCredentials: true })
      .then((response) => {
        console.log('API response data:', response.data);
        setAllStressData(response.data || []);
        filterDataByDate(response.data || [], new Date());
      })
      .catch((error) => {
        console.error('Error fetching stress data:', error);
      });
  }, []);

  // ✅ Don’t mutate the input date; use toDate for Mongo timestamps
  const filterDataByDate = (data, date) => {
    const start = new Date(date);
    start.setHours(0, 0, 0, 0);
    const end = new Date(date);
    end.setHours(23, 59, 59, 999);

    const filtered = (data || []).filter((item) => {
      const itemDate = toDate(item.timestamp);
      return itemDate >= start && itemDate <= end;
    });

    setFilteredStressData(filtered);
  };

  useEffect(() => {
    filterDataByDate(allStressData, selectedDate);
  }, [selectedDate, allStressData]);

  const lineChartData = {
    // ✅ Use toDate to handle {$date: "..."} and ensure valid Date objects
    labels: filteredStressData.map((d) => toDate(d.timestamp)),
    datasets: [
      {
        label: 'Stress Level Over Time',
        // ✅ Ensure numeric values
        data: filteredStressData.map((d) => Number(d.model_prediction)),
        fill: false,
        borderColor: '#1e90ff',
        tension: 0.3,
        pointBackgroundColor: filteredStressData.map((d) =>
          d.emotion === 'stressed' ? '#dc3545' : '#28a745'
        ),
        pointBorderColor: filteredStressData.map((d) =>
          d.emotion === 'stressed' ? '#dc3545' : '#28a745'
        ),
        pointRadius: 5,
        pointHoverRadius: 8,
      },
    ],
  };

  const lineChartOptions = {
    scales: {
      x: {
        type: 'time',
        time: {
          unit: 'hour',
          displayFormats: {
            hour: 'ha',
          },
          stepSize: 1,
        },
        title: {
          display: true,
          text: 'Time',
          color: '#ffffff',
        },
        ticks: {
          color: '#ffffff',
        },
        grid: {
          color: 'rgba(255,255,255,0.2)',
        },
      },
      y: {
        title: {
          display: true,
          text: 'Model Prediction Value',
          color: '#ffffff',
        },
        ticks: {
          color: '#ffffff',
        },
        grid: {
          color: 'rgba(255,255,255,0.2)',
        },
      },
    },
    plugins: {
      legend: {
        display: true,
        labels: {
          color: '#ffffff',
        },
      },
    },
    animation: {
      duration: 1500,
      easing: 'easeInOutQuad',
    },
  };

  const stressCounts = filteredStressData.reduce((acc, data) => {
    acc[data.emotion] = (acc[data.emotion] || 0) + 1;
    return acc;
  }, {});

  const totalEntries = filteredStressData.length;

  const pieChartData = {
    labels: ['Stressed', 'Not Stressed'],
    datasets: [
      {
        data: [stressCounts.stressed || 0, stressCounts['not stressed'] || 0],
        backgroundColor: ['#dc3545', '#28a745'],
        borderColor: ['#721c24', '#155724'],
        hoverOffset: 4,
      },
    ],
  };

  const pieChartOptions = {
    plugins: {
      legend: {
        display: true,
        labels: {
          color: '#ffffff',
        },
      },
      tooltip: {
        callbacks: {
          label: function (context) {
            let label = context.label || '';
            if (label) label += ': ';
            label += Math.round(context.raw * 100) / 100;
            return label;
          },
        },
      },
    },
    animation: {
      animateScale: true,
      animateRotate: true,
    },
  };

  return (
    <>
      <div className="dashboard-container">
        <div className="dashboard-content">
          <div className="charts-section">
            <div className="controls">
              <label style={{ marginTop: '5px', fontWeight: 'bold', color: 'black' }}>
                Select Date:{' '}
              </label>
              <DatePicker
                selected={selectedDate}
                onChange={(date) => setSelectedDate(date)}
                className="custom-datepicker form-control"
              />
            </div>
            <div className="row chart-wrapper">
              <div className="chart col-md-7">
                <h3 className="chart-title">Stress Level Over Time</h3>
                {filteredStressData.length > 0 ? (
                  <Line data={lineChartData} options={lineChartOptions} />
                ) : (
                  <p style={{ color: 'white' }}>No stress indicators found.</p>
                )}
              </div>
              <div className="pie-chart-container col-md-5">
                <h3 className="chart-title">Stress Summary</h3>
                {totalEntries > 0 ? (
                  <Pie data={pieChartData} options={pieChartOptions} />
                ) : (
                  <p style={{ color: 'white' }}>No data for pie chart.</p>
                )}
                <div className="pie-chart-info">
                  <h3 className="chart-title">Stress Data Summary</h3>
                  <ul style={{ color: 'white' }}>
                    <li>Total Entries: {totalEntries}</li>
                    <li>Stressed: {stressCounts.stressed || 0}</li>
                    <li>Not Stressed: {stressCounts['not stressed'] || 0}</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
          <StressManagementTips />
        </div>
      </div>
    </>
  );
};

export default Dashboard;
