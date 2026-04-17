import React, { useEffect, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Container,
  Box,
  Typography,
  Button,
  Paper,
  Stack,
  TextField,
  Chip,
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import Loading from '../../components/Loading/Loading';
import Toast from '../../components/Toast/Toast';
import { api } from '../../services/api';
import './Results.css';

function Results() {
  const navigate = useNavigate();
  const location = useLocation();
  const imageUrl = location.state?.imageUrl;

  const [loading, setLoading] = useState(true);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!imageUrl) {
      navigate('/', { replace: true });
      return;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);
    setResult(null);

    api
      .predictTags(imageUrl)
      .then((data) => {
        if (cancelled) return;
        setResult(data);
        setLoading(false);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err.message || 'Prediction failed');
        setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [imageUrl, navigate]);

  const tagEntries = result?.tags ? Object.entries(result.tags) : [];

  return (
    <Container maxWidth="lg" className="results-container">
      <Box className="results-header">
        <Button
          startIcon={<ArrowBackIcon />}
          color="primary"
          onClick={() => navigate('/')}
        >
          Back
        </Button>
        <Typography variant="h4" component="h1" className="results-title">
          Results
        </Typography>
      </Box>

      <Stack
        direction={{ xs: 'column', md: 'row' }}
        spacing={3}
        alignItems="stretch"
      >
        <Paper elevation={2} className="results-image-panel">
          {imageUrl && (
            <Box className="results-image-wrapper">
              <img src={imageUrl} alt="Input" className="results-image" />
              {tagEntries.length > 0 && (
                <>
                  <Box className="results-tags-column results-tags-left">
                    {tagEntries
                      .filter((_, i) => i % 2 === 0)
                      .map(([key, value], i) => (
                        <Chip
                          key={key}
                          label={`${key}: ${value}`}
                          size="small"
                          color="primary"
                          className="results-tag-chip"
                          style={{ animationDelay: `${i * 0.18}s` }}
                        />
                      ))}
                  </Box>
                  <Box className="results-tags-column results-tags-right">
                    {tagEntries
                      .filter((_, i) => i % 2 === 1)
                      .map(([key, value], i) => (
                        <Chip
                          key={key}
                          label={`${key}: ${value}`}
                          size="small"
                          color="primary"
                          className="results-tag-chip"
                          style={{ animationDelay: `${i * 0.18 + 0.09}s` }}
                        />
                      ))}
                  </Box>
                </>
              )}
            </Box>
          )}
        </Paper>

        <Paper elevation={2} className="results-panel">
          {loading ? (
            <Loading label="Running prediction..." />
          ) : result ? (
            <Stack spacing={2} className="results-fields">
              <Typography variant="h6" component="h2" className="results-section-title">
                Product Information
              </Typography>
              <TextField
                label="Title"
                value={result.title || ''}
                fullWidth
                InputProps={{ readOnly: true }}
              />
              <TextField
                label="Product Type"
                value={result.product_type || ''}
                fullWidth
                InputProps={{ readOnly: true }}
              />
              <TextField
                label="Description"
                value={result.description || ''}
                fullWidth
                multiline
                minRows={3}
                InputProps={{ readOnly: true }}
              />
            </Stack>
          ) : (
            <Typography variant="body2" color="text.secondary">
              No response.
            </Typography>
          )}
        </Paper>
      </Stack>

      <Toast
        open={!!error}
        message={error || ''}
        severity="error"
        onClose={() => setError(null)}
      />
    </Container>
  );
}

export default Results;
