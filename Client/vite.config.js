import basicSsl from '@vitejs/plugin-basic-ssl'

export default {
  server: {
    port: 5173, // change to the port you want
  },
  plugins: [
    basicSsl({
      /** name of certification */
      name: 'test',
      /** custom trust domains */
      domains: ['*'],
      /** custom certification directory */
    }),
  ],
}